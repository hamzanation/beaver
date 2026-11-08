import argparse
from transformers import set_seed
import os
from tqdm import tqdm, trange
import torch
import numpy as np
from typing import Literal
from pathlib import Path

from rerank import Reranker, Pair
from utils import format_table, read_json, write_json, cosine_sim

def get_torch_dtype(dtype: str, device: str):
    if dtype == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


def embed(
    texts: list[str],
    fn,
    model: str,
    provider: Literal['local', 'openrouter'],
    batch_size=4,
    device="cuda",
    dtype="auto",
):
    if fn is not None and os.path.isfile(fn):
        print(f'loading computed embedding from {fn}')
        return torch.from_numpy(np.load(fn))

    if provider == 'local':
        from sentence_transformers import SentenceTransformer
        torch_dtype = get_torch_dtype(dtype, device)
        model_kwargs = {"torch_dtype": torch_dtype}
        try:
            model = SentenceTransformer(
                model,
                trust_remote_code=True,
                device=device,
                model_kwargs=model_kwargs,
            )
        except TypeError:
            model = SentenceTransformer(model, trust_remote_code=True, device=device)
        embeds = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=batch_size,
            convert_to_numpy=True,
        )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    elif provider == 'openrouter':
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv()
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )

        batch_size = 256

        embeds = []
        for batch_idx in trange(0, len(texts), batch_size, desc="Batches"):
            batch_texts = texts[batch_idx : batch_idx + batch_size]
            response = client.embeddings.create(model=model, input=batch_texts)
            embeddings = [e.embedding for e in response.data]
            embeds += embeddings
        
        embeds = np.array(embeds)
    
    assert len(embeds) == len(texts)

    if fn is not None:
        np.save(fn, embeds)
    
    return torch.from_numpy(embeds)

def get_fns(dataset: str):
    dataset_dir = Path(f"./data/{dataset}")
    retrieval_dir = Path(f"./data/{dataset}/retrieval")
    retrieval_dir.mkdir(exist_ok=True)

    return {
        "qs_fn": dataset_dir / 'dev_sampled.json',
        "tables_fn": dataset_dir / 'dev_tables.json',
        "q_embeds_fn": retrieval_dir / 'q_embeds.npy',
        "t_embeds_fn": retrieval_dir / 't_embeds.npy',
        "retrieved_tables_fn": retrieval_dir / "retrieved_tables.json",
        "reranked_tables_fn": retrieval_dir / "reranked_tables.json"
    }

def retrieve_tables(dataset: str, model, k, provider, batch_size, device, dtype):
    fns = get_fns(dataset)

    # embed questions
    qs = read_json(fns['qs_fn'])
    q_embeds = embed([q['question'] for q in qs], fns['q_embeds_fn'], model, provider, batch_size, device, dtype)
    
    # embed tables
    tables = read_json(fns['tables_fn'])
    table_keys = list(tables.keys())
    trim = dataset in ['sp', 'nova']
    formatted_tables = [format_table(t, tables, use_instance=False, trim=trim) for t in tables]
    t_embeds = embed(formatted_tables, fns['t_embeds_fn'], model, provider, batch_size, device, dtype)

    print(f'computing cosine similarity between {len(q_embeds)} questions and {len(t_embeds)} tables')
    sim_scores = cosine_sim(q_embeds, t_embeds)

    top_k_indices = torch.topk(sim_scores, k=k, dim=1).indices
    top_k_tables = {qs[q_idx]['id']: [table_keys[i] for i in q_indices] for q_idx, q_indices in enumerate(top_k_indices)}
    write_json(top_k_tables, fns['retrieved_tables_fn'])
    print(f"Top-{k} retrieved tables saved to {fns['retrieved_tables_fn']}")

def rerank_tables(dataset: str, model, k, batch_size, device, dtype, max_length):
    fns = get_fns(dataset)

    qs, tables = read_json(fns['qs_fn']), read_json(fns['tables_fn'])
    trim = dataset in ['sp', 'nova']
    formatted_tables = {t: format_table(t, tables, use_instance=False, trim=trim) for t in tables}
    
    retrieved_tables = read_json(fns['retrieved_tables_fn'])
    reranked_tables = {}
    reranked_tables_fn = fns['reranked_tables_fn']
    if reranked_tables_fn.exists():
        reranked_tables = read_json(reranked_tables_fn)

    reranker = Reranker(model, device=device, dtype=dtype, max_length=max_length)

    for q in tqdm(qs):
        q_id = q['id']
        if q_id in reranked_tables:
            continue

        pairs = []

        for table_id in retrieved_tables[q_id]:
            pairs.append(Pair(q['question'], formatted_tables[table_id]))

        rerank_scores = []
        for batch_idx in range(0, len(pairs), batch_size):
            batch_pairs = pairs[batch_idx : batch_idx + batch_size]
            rerank_scores.append(reranker.rerank(batch_pairs))
        rerank_scores = torch.cat(rerank_scores)

        top_table_idxs = torch.sort(rerank_scores, descending=True).indices
        top_tables = [retrieved_tables[q_id][i] for i in top_table_idxs]
        reranked_tables[q_id] = top_tables[:k]
    
        write_json(reranked_tables, reranked_tables_fn)
    
    print(f"Top-{k} reranked tables saved to {reranked_tables_fn}")

def main():
    set_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--embed_model", type=str, default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--embed_k", type=int, default=50)
    parser.add_argument("--embed_provider", choices=['local', 'openrouter'])
    parser.add_argument("--embed_batch_size", type=int, default=4)
    parser.add_argument("--embed_device", choices=['cuda', 'cpu'], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embed_dtype", choices=['auto', 'float32', 'float16', 'bfloat16'], default="auto")
    parser.add_argument("--rerank_model", type=str, default=None)
    parser.add_argument("--rerank_k", type=int, default=15)
    parser.add_argument("--rerank_batch_size", type=int, default=1)
    parser.add_argument("--rerank_device", choices=['cuda', 'cpu'], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rerank_dtype", choices=['auto', 'float32', 'float16', 'bfloat16'], default="auto")
    parser.add_argument("--rerank_max_length", type=int, default=2048)
    args = parser.parse_args()

    print(f'Retrieving top-{args.embed_k} tables using {args.embed_model}...')
    retrieve_tables(
        args.dataset,
        args.embed_model,
        args.embed_k,
        args.embed_provider,
        args.embed_batch_size,
        args.embed_device,
        args.embed_dtype,
    )

    if args.rerank_model:
        print(f'\nReranking top-{args.rerank_k} tables using {args.rerank_model}...')
        rerank_tables(
            args.dataset,
            args.rerank_model,
            args.rerank_k,
            args.rerank_batch_size,
            args.rerank_device,
            args.rerank_dtype,
            args.rerank_max_length,
        )

if __name__ == "__main__":
    main()
