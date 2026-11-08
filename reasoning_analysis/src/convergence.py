"""
Answer convergence: do models that reach the SAME answer reason the same way, and
do models that fail fail the SAME way?

Re-executes every model's predicted SQL (and the gold SQL) against the BEAVER MySQL
database, then -- per instance -- groups the models' predictions into result-set
equivalence classes using the same comparison the benchmark scores with
(`eval/utils/ex_acc.compare_results`: any column permutation, set of rows, column
names/duplicates ignored). Two models "agree" when their results are equivalent,
whether or not either matches gold.

Per (instance, model pair) it then joins that agreement label to three
reasoning-similarity measures, all within-instance (the instance's content is held
fixed, so no residualization is needed):
  - trace_cosine: cosine of the raw mean-pooled bge-large trace embeddings
    (Phase 2's stored vectors -- only the 47 instances all 6 models attempted);
  - move_js: Jensen-Shannon divergence between the two traces' move-share vectors
    (Phase 2F's per-sentence labels);
  - table_jaccard: Jaccard overlap of the tables referenced in the two final SQLs
    (a cheap "same plan?" signal that is about the answer, not the trace).
"""
import html
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from . import data_loading as dl
from .move_taxonomy import MOVES

sys.path.insert(0, str(dl.REPO_ROOT / "eval"))
from utils.ex_acc import compare_results, execute_sql_with_timeout, get_mysql_credentials  # noqa: E402

_TABLE_RE = re.compile(r"\b(?:FROM|JOIN)\s+`?([A-Za-z_][A-Za-z0-9_]*)`?", re.IGNORECASE)


def _mysql_creds():
    try:
        from dotenv import load_dotenv
        load_dotenv(dl.REPO_ROOT / ".env")
    except ImportError:
        pass
    creds = get_mysql_credentials("dw")
    if not creds:
        raise SystemExit("No MySQL credentials (set MYSQL_* in the repo-root .env).")
    return creds


def referenced_tables(sql):
    """Physical tables named after FROM/JOIN, minus CTE names defined in the query."""
    if not sql:
        return frozenset()
    ctes = {m.lower() for m in re.findall(r"(?:WITH|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(", sql, re.IGNORECASE)}
    return frozenset(t.upper() for t in _TABLE_RE.findall(sql) if t.lower() not in ctes)


def _read(path):
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return text or None


def execute_all(master, unescape=False):
    """Executes gold + every prediction once. `unescape` html-unescapes predictions
    first: the GPT models write `&gt;`/`&lt;` inside their <ans> tags, which MySQL
    rejects, so as-scored those queries never execute. Returns (results, sql_of,
    outcome_rows); outcome_rows' `correct` is re-scored from these executions:
    results[(model or 'gold', iid)] = DataFrame or None; outcome_rows = one dict per
    (model, iid) describing how the prediction relates to gold."""
    creds = _mysql_creds()
    run_of = master.drop_duplicates("model").set_index("model")["run_name"].to_dict()
    unified = dl.FEWSHOT_UNIFIED

    results, sql_of, err_of = {}, {}, {}
    iids = sorted(master["instance_id"].unique())
    any_run = next(iter(run_of.values()))
    for iid in iids:
        gold_sql = _read(unified / any_run / "gold" / f"{iid}.sql")
        df, err = execute_sql_with_timeout(gold_sql, creds) if gold_sql else (None, "no gold")
        results[("gold", iid)], err_of[("gold", iid)] = df, err

    for r in master.itertuples():
        key = (r.model, r.instance_id)
        sql = _read(unified / run_of[r.model] / "generated" / f"{r.instance_id}.sql")
        if sql is not None and unescape:
            sql = html.unescape(sql)
        sql_of[key] = sql
        if sql is None:
            results[key], err_of[key] = None, "no prediction"
        else:
            results[key], err_of[key] = execute_sql_with_timeout(sql, creds)
        print(f"  executed {r.model} {r.instance_id}: {'ERR ' + err_of[key][:60] if err_of[key] else 'ok'}")

    outcome_rows = []
    for r in master.itertuples():
        key = (r.model, r.instance_id)
        pred, gold = results[key], results[("gold", r.instance_id)]
        correct = equivalent(pred, gold)
        outcome_rows.append({
            "model": r.model,
            "instance_id": r.instance_id,
            "correct": correct,
            "official_correct": bool(r.correct),
            "has_reasoning_trace": bool(getattr(r, "has_reasoning_trace", True)),
            "had_html_entities": bool(sql_of[key]) and sql_of[key] != _read(
                unified / run_of[r.model] / "generated" / f"{r.instance_id}.sql"),
            "error_type": classify_error(pred, err_of[key], gold, correct),
            "pred_rows": None if pred is None else len(pred),
            "pred_cols": None if pred is None else pred.shape[1],
            "gold_rows": None if gold is None else len(gold),
            "gold_cols": None if gold is None else gold.shape[1],
            "tables": ";".join(sorted(referenced_tables(sql_of[key]))),
            "exec_error": err_of[key],
        })
    return results, sql_of, outcome_rows


def classify_error(pred, err, gold, correct):
    if correct:
        return "correct"
    if err == "no prediction":
        return "no_prediction"
    if err:
        return "timeout" if "Timeout" in err else "exec_error"
    if pred is None or pred.empty:
        return "empty_result"
    if gold is not None and pred.shape[1] != gold.shape[1]:
        return "wrong_columns"
    if gold is not None and len(pred.drop_duplicates()) != len(gold.drop_duplicates()):
        return "wrong_row_count"
    return "wrong_values"


def equivalent(a, b):
    """Result-set equivalence between two predictions. Failed/missing executions are
    never equivalent to anything (two different crashes are not 'the same answer')."""
    if a is None or b is None:
        return False
    try:
        return bool(compare_results(a, b)[0])
    except Exception:
        return False


def answer_clusters(results, models, iid):
    """Union-find over models whose results are equivalent. Returns model -> cluster id
    (0 = the gold-matching cluster if any model is correct, else ordered by size)."""
    parent = {m: m for m in models}

    def find(m):
        while parent[m] != m:
            parent[m] = parent[parent[m]]
            m = parent[m]
        return m

    for a, b in combinations(models, 2):
        if equivalent(results[(a, iid)], results[(b, iid)]):
            parent[find(a)] = find(b)
    groups = {}
    for m in models:
        groups.setdefault(find(m), []).append(m)
    gold = results[("gold", iid)]
    ordered = sorted(groups.values(), key=lambda g: (not equivalent(results[(g[0], iid)], gold), -len(g)))
    return {m: i for i, g in enumerate(ordered) for m in g}


def _js(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.sum() == 0 or q.sum() == 0:
        return np.nan
    p, q = p / p.sum(), q / q.sum()
    mix = (p + q) / 2

    def kl(x, y):
        nz = x > 0
        return float(np.sum(x[nz] * np.log2(x[nz] / y[nz])))
    return 0.5 * kl(p, mix) + 0.5 * kl(q, mix)


def move_shares(move_labels):
    out = {}
    for (m, iid), g in move_labels.groupby(["model", "instance_id"]):
        c = Counter(g["predicted_move"])
        out[(m, iid)] = np.array([c.get(mv, 0) for mv in MOVES], float)
    return out


def pair_table(master, results, sql_of, trace_emb, shares):
    """One row per (instance, model pair), both models attempted the instance."""
    rows = []
    cluster_rows = []
    for iid, g in master.groupby("instance_id"):
        models = sorted(g["model"])
        correct = g.set_index("model")["correct"].to_dict()
        clusters = answer_clusters(results, models, iid)
        n_correct = sum(correct.values())
        for m in models:
            cluster_rows.append({"instance_id": iid, "model": m, "correct": correct[m],
                                 "cluster": clusters[m], "n_models": len(models),
                                 "n_correct": n_correct})
        for a, b in combinations(models, 2):
            ea, eb = trace_emb.get((a, iid)), trace_emb.get((b, iid))
            cos = float(ea @ eb / (np.linalg.norm(ea) * np.linalg.norm(eb))) if ea is not None and eb is not None else np.nan
            sa, sb = shares.get((a, iid)), shares.get((b, iid))
            ta, tb = referenced_tables(sql_of[(a, iid)]), referenced_tables(sql_of[(b, iid)])
            same = clusters[a] == clusters[b] and equivalent(results[(a, iid)], results[(b, iid)])
            if correct[a] and correct[b]:
                kind = "both_correct"
            elif correct[a] != correct[b]:
                kind = "one_correct"
            else:
                kind = "both_wrong_same_answer" if same else "both_wrong_diff_answer"
            rows.append({
                "instance_id": iid, "model_a": a, "model_b": b,
                "same_brand": dl.brand_of(a) == dl.brand_of(b),
                "n_models_attempted": len(models), "instance_n_correct": n_correct,
                "pair_kind": kind, "same_answer": same,
                "trace_cosine": cos,
                "move_js": _js(sa, sb) if sa is not None and sb is not None else np.nan,
                "table_jaccard": (len(ta & tb) / len(ta | tb)) if (ta | tb) else np.nan,
            })
    return pd.DataFrame(rows), pd.DataFrame(cluster_rows)


def within_instance_contrast(pairs, metric, kind_a, kind_b):
    """Mean(metric | kind_a) - mean(metric | kind_b), computed WITHIN each instance that
    has both kinds of pair, then averaged across instances; CI by bootstrapping
    instances. Holding the instance fixed removes 'easy problems have similar traces'."""
    diffs = []
    for _iid, g in pairs.dropna(subset=[metric]).groupby("instance_id"):
        xa, xb = g.loc[g.pair_kind.isin(kind_a), metric], g.loc[g.pair_kind.isin(kind_b), metric]
        if len(xa) and len(xb):
            diffs.append(xa.mean() - xb.mean())
    diffs = np.array(diffs)
    if len(diffs) == 0:
        return {"n_instances": 0}
    rng = np.random.default_rng(0)
    boots = [rng.choice(diffs, len(diffs)).mean() for _ in range(2000)]
    return {"n_instances": len(diffs), "mean_diff": diffs.mean(),
            "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5),
            "share_positive": float((diffs > 0).mean())}
