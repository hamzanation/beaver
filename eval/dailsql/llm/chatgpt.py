import json.decoder
from openai import OpenAI
from utils.enums import LLM
import time
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from local_llm import is_lmstudio_model, lmstudio_client, lmstudio_model_name, response_usage
from anthropic_llm import anthropic_chat, is_anthropic_model
from usage_utils import new_usage

CLIENT = None

def init_chatgpt(OPENAI_API_KEY, OPENAI_GROUP_ID, model):
    global CLIENT
    if is_lmstudio_model(model):
        CLIENT = lmstudio_client()
    elif is_anthropic_model(model):
        CLIENT = None
    elif "gemini" in model:
        CLIENT = OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    elif "/" in model:
        CLIENT = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
    else:
        CLIENT = OpenAI(api_key=OPENAI_API_KEY)
        # openai.organization = OPENAI_GROUP_ID # v1.x handles this differently if needed, usually just api_key is enough for standard usage

def ask_chat(model, messages: list, temperature, n, max_tokens):
    global CLIENT
    # print('>>>model:', model)
    # print('>>>max_tokens:', max_tokens)
    
    # Ensure client is initialized
    if CLIENT is None and not is_anthropic_model(model):
        raise RuntimeError("OpenAI Client not initialized. Call init_chatgpt first.")

    api_model = lmstudio_model_name(model) if is_lmstudio_model(model) else model
    t0 = time.time()

    if is_anthropic_model(model):
        thinking_budget = os.environ.get("BEAVER_THINKING_BUDGET_TOKENS")
        thinking_effort = os.environ.get("BEAVER_THINKING_EFFORT")
        response = anthropic_chat(
            model, messages, temperature=temperature, n=n, max_tokens=max_tokens,
            thinking_budget_tokens=int(thinking_budget) if thinking_budget else None,
            thinking_effort=thinking_effort,
        )
        response["latency_sec"] = time.time() - t0
        return response
    elif is_lmstudio_model(model):
        response = CLIENT.chat.completions.create(
            model=api_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n
        )
    elif "gemini" in model:
         # Gemini logic
        try:
            if "2" in model:
                response = CLIENT.chat.completions.create(
                    model=api_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                    # reasoning_effort="none"
                )
            else:
             response = CLIENT.chat.completions.create(
                model=api_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                reasoning_effort="none"
            )
        except Exception as e:
            # Fallback if reasoning_effort is not supported or other error, retry without it
            print(f"Gemini call failed with reasoning_effort='none', retrying without it. Error: {e}")
            response = CLIENT.chat.completions.create(
                model=api_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n
            )
    elif "/" in model:
        if "qwen" in model or "claude" in model:
            response = CLIENT.chat.completions.create(
                model=api_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                extra_body={
                    "reasoning": {
                        "effort": "none",
                    }
                },
        )
        elif "minimax" in model:
            response = CLIENT.chat.completions.create(
                model=api_model,
                messages=messages,
                temperature=temperature,
                # max_tokens=max_tokens,
                n=n
        )

    elif "gpt-5" in model or "o4-mini" in model:
        kwargs = dict(
            model=api_model,
            messages=messages,
            # temperature=temperature,
            # max_completion_tokens=max_tokens,
            n=n,
        )
        reasoning_effort = os.environ.get("BEAVER_REASONING_EFFORT")
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        response = CLIENT.chat.completions.create(**kwargs)
    elif model == LLM.GPT_o1:
        response = CLIENT.chat.completions.create(
            model=api_model,
            messages=messages,
            # temperature=temperature, # o1 doesn't support temp
            # max_completion_tokens=max_tokens,
            n=n
        )
    else:
        response = CLIENT.chat.completions.create(
            model=api_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n
        )
    
    response_clean = [choice.message.content for choice in response.choices]
    return dict(
        response=response_clean,
        latency_sec=time.time() - t0,
        # usage might be different object in v1, access safely
        **response_usage(response)
    )


def ask_llm(model: str, batch: list, temperature: float, n:int, max_tokens):
    try:
        # assert model in LLM.TASK_CHAT, "LLM.TASK_COMPLETIONS is too old. not supported."
        # batch size must be 1
        assert len(batch) == 1, "batch must be 1 in this mode"
        messages = [{"role": "user", "content": batch[0]}]
        response = ask_chat(model, messages, temperature, n, max_tokens)
        response['response'] = [response['response']]  # hard-code for batch_size=1
    except Exception as e:
        print(f"Error occurred in ask_llm: {e}")
        # import traceback
        # traceback.print_exc()
        # Return hard-coded response
        print("Return hard-coded response")
        response = {**new_usage(), "response": [["SELECT" for _ in range(n)]], "latency_sec": 0.0, "error": True}

    return response

