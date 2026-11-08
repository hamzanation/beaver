import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

from prompt import get_ete_prompts
from utils import EvalConfig

import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_llm import (
    is_lmstudio_model, lmstudio_client, lmstudio_model_name, response_usage,
    response_usage_responses_api, extract_reasoning_summary_text,
)
from anthropic_llm import anthropic_chat_once, is_anthropic_model
from usage_utils import (
    new_usage_record, record_call, finalize_cost, write_usage_json,
    new_reasoning_trace, record_reasoning_trace, write_reasoning_trace_json,
)
from pricing import PRICING
from sql_sanitize import sanitize_sql

MODEL_MAP = {
    "gpt-5-mini": "gpt-5-mini",
    "gpt-5.2": "gpt-5.2",
    "o4-mini": "o4-mini",
    "gemini": "gemini-2.0-flash",
    "qwen": "qwen/qwen3-next-80b-a3b-instruct",
    "claude": "claude",
    "minimax": "minimax/minimax-m2.1",
}


def process_output(output):
    # remove everything of left of <ans> if <ans> is in output
    if "<ans>" in output:
        output = output.split("<ans>")[-1]
    # remove everything of right of </ans> if </ans> is in output
    if "</ans>" in output:
        output = output.split("</ans>")[0]
    # remove everything of left of ```sql if ```sql is in output
    if "```sql" in output:
        output = output.split("```sql")[-1]
    # remove everything of right of ``` if ``` is in output
    if "```" in output:
        output = output.split("```")[0]
    # unescape HTML entities (&lt; / &gt;) and map the "None" abstention to "";
    # the raw response is still preserved verbatim in generation.log
    return sanitize_sql(output)




def run_prompts(prompts, indices, instance_ids, output_dir: str, model: str, dataset: str = None, setting: int = None):
    prompts_to_process = []
    indices_to_process = []
    instance_ids_to_process = []
    
    for prompt, idx, instance_id in zip(prompts, indices, instance_ids):
        instance_dir = os.path.join(output_dir, instance_id)
        sql_file = os.path.join(instance_dir, "predicted_0.sql")
        
        # Check if already processed
        if not os.path.exists(sql_file) or os.path.getsize(sql_file) == 0:
            prompts_to_process.append(prompt)
            indices_to_process.append(idx)
            instance_ids_to_process.append(instance_id)
            
    if not prompts_to_process:
        print("All prompts already processed.")
        return

    print(f"Running {len(prompts_to_process)} prompts...")
    
    client = None
    model_name = MODEL_MAP.get(model, model)


    use_lmstudio = is_lmstudio_model(model_name)
    if use_lmstudio:
        client = lmstudio_client()
        model_name = lmstudio_model_name(model_name)
    elif "gemini" in model_name:
        client = OpenAI(
                    api_key=os.environ.get("GEMINI_API_KEY"),
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )
    elif "qwen" in model_name:
        client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ.get("OPENROUTER_API_KEY"),
                )
    elif is_anthropic_model(model_name):
        client = None
    elif "minimax" in model_name:
        client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ.get("OPENROUTER_API_KEY"),
                )
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def process_prompt(item):
        prompt, idx, instance_id = item
        if prompt is None:
            return idx, instance_id, "", "", None

        t0 = time.time()
        try:
            if use_lmstudio:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=prompt,
                    temperature=0,
                    max_tokens=int(os.environ.get("BEAVER_MAX_TOKENS", "1024")),
                )
            elif "gemini" in model_name:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=prompt,
                    # temperature=0,
                )
            elif "qwen" in model_name:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=prompt,
                    # temperature=0,
                    extra_body={
                    "reasoning": {
                        "effort": "none",
                    }
                },
                )
            elif is_anthropic_model(model_name):
                thinking_budget = os.environ.get("BEAVER_THINKING_BUDGET_TOKENS")
                thinking_effort = os.environ.get("BEAVER_THINKING_EFFORT")
                result = anthropic_chat_once(
                    model_name,
                    prompt,
                    temperature=0,
                    max_tokens=int(os.environ.get("BEAVER_MAX_TOKENS", "1024")),
                    thinking_budget_tokens=int(thinking_budget) if thinking_budget else None,
                    thinking_effort=thinking_effort,
                )
                latency_sec = time.time() - t0
                content = result["content"]
                generation_log = f"Prompt:\n{prompt}\n\nResponse:\n{content}"
                usage = dict(result["usage"])
                if result.get("stop_reason") is not None:
                    usage["stop_reason"] = result["stop_reason"]
                if result.get("block_types_seen") is not None:
                    usage["block_types_seen"] = result["block_types_seen"]
                if result.get("raw_non_text_blocks_debug") is not None:
                    usage["raw_non_text_blocks_debug"] = result["raw_non_text_blocks_debug"]
                return idx, instance_id, content, generation_log, {
                    "usage": usage,
                    "latency_sec": latency_sec,
                    "thinking_text": result.get("thinking_text"),
                }
            elif "minimax" in model_name:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=prompt,
                    # temperature=0,
                )
            else:
                reasoning_effort = os.environ.get("BEAVER_REASONING_EFFORT")
                is_reasoning_model = "gpt-5" in model_name or "o3" in model_name or "o4-mini" in model_name
                if reasoning_effort and is_reasoning_model:
                    # Chat Completions never exposes reasoning content, only a token
                    # count -- the Responses API is the only way to get a (summarized,
                    # not raw) look at the reasoning itself.
                    resp = client.responses.create(
                        model=model_name,
                        input=prompt,
                        reasoning={"effort": reasoning_effort, "summary": "auto"},
                    )
                    latency_sec = time.time() - t0
                    content = resp.output_text
                    generation_log = f"Prompt:\n{prompt}\n\nResponse:\n{content}"
                    return idx, instance_id, content, generation_log, {
                        "usage": response_usage_responses_api(resp),
                        "latency_sec": latency_sec,
                        "thinking_text": extract_reasoning_summary_text(resp),
                    }
                resp = client.chat.completions.create(model=model_name, messages=prompt)

            latency_sec = time.time() - t0
            content = resp.choices[0].message.content
            # To match the structure of other baselines, we could log the prompt/response
            # Here we provide a simple generic generation log.
            generation_log = f"Prompt:\n{prompt}\n\nResponse:\n{content}"
            return idx, instance_id, content, generation_log, {"usage": response_usage(resp), "latency_sec": latency_sec}
        except Exception as e:
            latency_sec = time.time() - t0
            print(f"Error processing index {idx}: {e}")
            return idx, instance_id, "", f"Error: {e}", {"usage": None, "latency_sec": latency_sec, "error": True}

    default_workers = "1" if use_lmstudio or is_anthropic_model(model_name) else "20"
    max_workers = int(os.environ.get("BEAVER_NUM_WORKERS", default_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Zip prompts with indices for processing
        items = list(zip(prompts_to_process, indices_to_process, instance_ids_to_process))
        
        # Map returns results in the same order as input
        results = list(tqdm(executor.map(process_prompt, items), total=len(items)))

        # save results
        for idx, instance_id, content, gen_log, call_info in results:
            processed_sql = process_output(content)

            instance_dir = os.path.join(output_dir, instance_id)
            os.makedirs(instance_dir, exist_ok=True)

            sql_file = os.path.join(instance_dir, "predicted_0.sql")
            log_file = os.path.join(instance_dir, "generation.log")

            with open(sql_file, "w", encoding="utf-8") as f:
                f.write(processed_sql)
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(gen_log)

            if call_info is not None:
                usage_record = new_usage_record(instance_id, model, dataset=dataset, setting=setting)
                record_call(
                    usage_record,
                    stage="generation",
                    usage=call_info.get("usage"),
                    latency_sec=call_info.get("latency_sec", 0.0),
                    error=call_info.get("error", False),
                )
                finalize_cost(usage_record, PRICING)
                write_usage_json(os.path.join(instance_dir, "usage.json"), usage_record)

                thinking_text = call_info.get("thinking_text")
                if thinking_text:
                    trace = new_reasoning_trace(instance_id, model, dataset=dataset, setting=setting)
                    record_reasoning_trace(trace, "generation", thinking_text)
                    write_reasoning_trace_json(os.path.join(instance_dir, "reasoning_trace.json"), trace)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--dataset", type=str, default="dw")
    parser.add_argument("--setting", type=int, default=None, help="Hint setting (0/1/2), recorded in per-example usage.json only")
    parser.add_argument("--q_fn", type=str, default="dev")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="../../data")
    parser.add_argument("--gold_tables", action="store_true")
    parser.add_argument("--join_keys", action="store_true")
    parser.add_argument("--mapping", action="store_true")
    parser.add_argument("--knowledge", action="store_true")
    parser.add_argument("--decomp", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N questions (cheap smoke tests instead of the full sample)")
    parser.add_argument("--ids_file", type=str, default=None, help="JSON list of instance ids (or an object with an 'ids' list) to restrict the run to, e.g. eval/pretest/pretest_ids.json")
    args = parser.parse_args()

    model = args.model
    dataset = args.dataset
    q_fn = args.q_fn

    eval_config = EvalConfig(
        gold_tables=args.gold_tables, join_keys=args.join_keys, mapping=args.mapping, knowledge=args.knowledge, decomp=args.decomp
    )
    print(eval_config)

    prompts, instance_ids = get_ete_prompts(dataset, q_fn, eval_config, args.data_dir)
    if args.ids_file:
        import json
        with open(args.ids_file, "r", encoding="utf-8") as f:
            wanted = json.load(f)
        wanted = set(wanted["ids"] if isinstance(wanted, dict) else wanted)
        missing = wanted - set(instance_ids)
        if missing:
            raise SystemExit(f"--ids_file contains ids not in {q_fn}: {sorted(missing)}")
        keep = [i for i, iid in enumerate(instance_ids) if iid in wanted]
        prompts = [prompts[i] for i in keep]
        instance_ids = [instance_ids[i] for i in keep]
        print(f"Restricted to {len(instance_ids)} ids from {args.ids_file}")
    if args.limit:
        prompts = prompts[:args.limit]
        instance_ids = instance_ids[:args.limit]

    output_dir = args.output_dir

    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    selected_indices = list(range(len(prompts)))
    selected_prompts = [prompts[i] for i in selected_indices]

    run_prompts(selected_prompts, selected_indices, instance_ids, output_dir, model, dataset=dataset, setting=args.setting)
