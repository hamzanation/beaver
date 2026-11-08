"""
Phase 2 analyses A-C: residualization, within-instance model-by-model similarity +
hierarchical clustering, and brand/size_tier probes.

Scope note (Phase 0 decision, applied consistently through Phase 2): cross-model
comparisons here are restricted to the fully-overlapping instance set (all 6 models
have a trace) -- residualizing or comparing across a partial, per-instance-varying
set of models would mix apples and oranges (a 2-model mean vs. a 6-model mean).
D/E (UMAP, trajectory metrics) and F/G (move taxonomy, BERTopic) are not in this file yet.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, permutation_test_score
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

from . import embeddings as emb
from . import text_normalize as tn


def compute_trace_embeddings(rows, model_name=emb.DEFAULT_SENTENCE_MODEL, device="cuda", verbose=True):
    """(model, instance_id) -> mean-pooled sentence embedding, for every row with a
    reasoning trace. Sentences are split from the NORMALIZED text (headers/progress
    markers stripped) so sentence boundaries aren't polluted by markdown."""
    out = {}
    n = sum(1 for r in rows if r["has_reasoning_trace"])
    done = 0
    for r in rows:
        if not r["has_reasoning_trace"]:
            continue
        normalized = tn.normalize_text(r["raw_text"])
        sentences = tn.split_sentences(normalized)
        vec = emb.embed_trace_mean_pooled(sentences, model_name=model_name, device=device)
        if vec is not None:
            out[(r["model"], r["instance_id"])] = vec
        done += 1
        if verbose and done % 100 == 0:
            print(f"    trace embeddings: {done}/{n}")
    return out


def compute_long_context_embeddings(rows, model_name=emb.LONG_CONTEXT_MODEL, device="cuda", verbose=True):
    """(model, instance_id) -> whole-trace embedding via the long-context model, as
    the robustness check against mean-pooled sentence embeddings."""
    out = {}
    n = sum(1 for r in rows if r["has_reasoning_trace"])
    done = 0
    for r in rows:
        if not r["has_reasoning_trace"]:
            continue
        normalized = tn.normalize_text(r["raw_text"])
        vec = emb.embed_trace_long_context(normalized, model_name=model_name, device=device)
        if vec is not None:
            out[(r["model"], r["instance_id"])] = vec
        done += 1
        if verbose and done % 100 == 0:
            print(f"    long-context embeddings: {done}/{n}")
    return out


def residualize(trace_embeddings: dict, shared_ids: set, models: list):
    """For each shared instance_id, subtract the mean embedding across `models` from
    each model's embedding. Returns (residuals, raw_subset), both dict[(model,iid)] -> vec,
    restricted to instances where ALL given models have an embedding."""
    raw_subset = {}
    residuals = {}
    for iid in shared_ids:
        vecs = {}
        for m in models:
            v = trace_embeddings.get((m, iid))
            if v is None:
                break
            vecs[m] = v
        if len(vecs) != len(models):
            continue  # not all models present for this instance -- skip (shouldn't happen given shared_ids, but be safe)
        mean_vec = np.mean(list(vecs.values()), axis=0)
        for m, v in vecs.items():
            raw_subset[(m, iid)] = v
            residuals[(m, iid)] = v - mean_vec
    return residuals, raw_subset


def model_similarity_matrix(trace_embeddings: dict, shared_ids: set, models: list):
    """Average cosine similarity between every model pair, over shared instances
    (embeddings are pre-normalized, so cosine similarity == dot product). Also returns
    a scipy linkage matrix from the corresponding distance matrix for dendrogram use."""
    n = len(models)
    sim_sum = np.zeros((n, n))
    counts = np.zeros((n, n))
    for iid in shared_ids:
        vecs = [trace_embeddings.get((m, iid)) for m in models]
        if any(v is None for v in vecs):
            continue
        for i in range(n):
            for j in range(n):
                sim_sum[i, j] += float(np.dot(vecs[i], vecs[j]))
                counts[i, j] += 1

    with np.errstate(invalid="ignore"):
        sim_matrix = np.divide(sim_sum, counts, out=np.full_like(sim_sum, np.nan), where=counts > 0)

    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0  # symmetrize away float noise
    condensed = squareform(dist_matrix, checks=False)
    link = linkage(condensed, method="average")

    return {"models": models, "similarity_matrix": sim_matrix, "distance_matrix": dist_matrix,
            "linkage": link, "n_shared_instances": len(shared_ids)}


def _majority_baseline(y):
    from collections import Counter
    counts = Counter(y)
    return max(counts.values()) / len(y)


def run_probe(X, y, groups, n_permutations=200, random_state=emb.SEED, standardize=True):
    """LogisticRegression + GroupKFold(instance_id) + permutation test, vs. a
    majority-class baseline. Returns accuracy, baseline, p-value, and the fitted
    coefficients (only meaningful when X's columns are interpretable, e.g. lexical
    features -- caller decides whether to report them)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    groups = np.asarray(groups)

    n_groups = len(set(groups.tolist()))
    n_splits = min(5, n_groups)
    if n_splits < 2:
        return {"error": f"only {n_groups} distinct instance_id groups, need >=2 for GroupKFold"}

    if standardize:
        X = StandardScaler().fit_transform(X)

    clf = LogisticRegression(max_iter=2000, random_state=random_state)
    cv = GroupKFold(n_splits=n_splits)

    score, perm_scores, pvalue = permutation_test_score(
        clf, X, y, groups=groups, cv=cv, n_permutations=n_permutations,
        random_state=random_state, scoring="accuracy", n_jobs=-1,
    )

    clf.fit(X, y)
    baseline = max(_majority_baseline(y), DummyClassifier(strategy="most_frequent").fit(X, y).score(X, y))

    return {
        "cv_accuracy": score,
        "majority_baseline": baseline,
        "permutation_pvalue": pvalue,
        "n_permutations": n_permutations,
        "n_samples": len(y),
        "n_groups": n_groups,
        "coefficients": clf.coef_,
        "classes": clf.classes_,
    }


def run_umap(residuals: dict, n_neighbors=15, min_dist=0.1, random_state=emb.SEED):
    """UMAP of residual (style, content-removed) embeddings. Returns (pairs, coords)
    where pairs[i] = (model, instance_id) for coords[i]."""
    import umap
    pairs = sorted(residuals.keys())
    X = np.stack([residuals[p] for p in pairs])
    n_neighbors = min(n_neighbors, max(2, len(pairs) - 1))
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state, metric="cosine")
    coords = reducer.fit_transform(X)
    return pairs, coords


def top_lexical_features(coefficients, classes, feature_names, top_k=15):
    """Sorted by |coefficient| for a binary probe, or by max-abs-across-classes for
    multiclass (one row of coefficients per class for multinomial LogisticRegression)."""
    coefs = np.asarray(coefficients)
    if coefs.ndim == 1 or coefs.shape[0] == 1:
        row = coefs.reshape(-1)
        order = np.argsort(-np.abs(row))[:top_k]
        return [{"feature": feature_names[i], "coefficient": float(row[i])} for i in order]
    else:
        max_abs = np.max(np.abs(coefs), axis=0)
        order = np.argsort(-max_abs)[:top_k]
        rows = []
        for i in order:
            per_class = {str(classes[c]): float(coefs[c, i]) for c in range(coefs.shape[0])}
            rows.append({"feature": feature_names[i], **per_class})
        return rows
