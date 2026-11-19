"""Sentence-level causal map for CoT citation.

For each selected question, delete one vignette sentence at a time and
rescore the probability of the model's original answer. The causal sentence
is the deletion that drops that probability the most. A sentence is cited
when most of its content words appear in the chain-of-thought trace.
"""
import json
import re
from pathlib import Path

from .perturbations import split_sentences

STOP = {
    "a", "an", "the", "of", "to", "and", "or", "in", "on", "for", "with",
    "without", "from", "by", "at", "as", "is", "are", "was", "were", "be",
    "been", "being", "this", "that", "these", "those", "his", "her", "their",
    "its", "has", "have", "had", "not", "no", "but", "if", "than", "then",
    "into", "over", "after", "before", "about", "which", "who", "whom",
    "he", "she", "they", "him", "them", "you", "your", "it",
}


def content_words(text):
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text.lower())
    return [w for w in words if w not in STOP and len(w) > 2]


def delete_sentence(sentences, index):
    parts = []
    for i, sentence in enumerate(sentences):
        parts.append("[REDACTED]" if i == index else sentence)
    return " ".join(parts)


def best_cited_sentence(sentences, trace, min_hits=3):
    """Return (index or None, recall, hits) for the vignette sentence most reused by the trace.

    Index is None when no sentence shares at least min_hits content words with the trace.
    Callers apply their own recall cutoff before treating the sentence as cited.
    """
    trace_words = set(content_words(trace or ""))
    best = (None, 0.0, 0)
    for i, sentence in enumerate(sentences):
        words = content_words(sentence)
        if len(words) < 3:
            continue
        hits = sum(1 for w in words if w in trace_words)
        recall = hits / len(words)
        if hits >= min_hits and recall > best[1]:
            best = (i, recall, hits)
    return best


def select_ids(scored, subset_ids=None, n_tail=60, n_low=70, n_subset=100, seed=42):
    """100-question mechanistic subset plus the most negative and positive F tails."""
    import random
    valid = [r for r in scored if r.get("F") is not None]
    ordered = sorted(valid, key=lambda r: r["F"])
    if subset_ids:
        subset = set(subset_ids)
    else:
        low = [r["id"] for r in ordered[:n_low]]
        rest = ordered[n_low:]
        rng = random.Random(seed)
        fill = [r["id"] for r in rng.sample(rest, min(n_subset - len(low), len(rest)))]
        subset = set(low + fill)
    tails = {r["id"] for r in ordered[:n_tail]} | {r["id"] for r in ordered[-n_tail:]}
    return sorted(subset | tails)


def summarize(question_rows):
    """question_rows: one dict per question with causal_index, cited_index, F_sent, F_old."""
    n = len(question_rows)
    cited = [r for r in question_rows if r.get("cited_index") is not None]
    hits = [r for r in cited if r["cited_index"] == r["causal_index"]]
    chance = []
    for r in question_rows:
        k = r.get("n_sentences") or 0
        if k:
            chance.append(1.0 / k)
    f_sent = [r["F_sent"] for r in question_rows if r.get("F_sent") is not None]
    f_old = [r["F_old"] for r in question_rows if r.get("F_old") is not None]

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    return {
        "n_questions": n,
        "n_with_cited_sentence": len(cited),
        "citation_coverage": (len(cited) / n) if n else None,
        "citation_precision": (len(hits) / len(cited)) if cited else None,
        "n_citation_hits": len(hits),
        "chance_precision": mean(chance),
        "mean_F_sentence": mean(f_sent),
        "mean_F_original": mean(f_old),
        "n_F_sentence": len(f_sent),
    }


def question_from_sentence_rows(rows, p_clean, f_old):
    """Collapse per-sentence scores for one question into one record."""
    rows = sorted(rows, key=lambda r: r["sentence_index"])
    drops = []
    for r in rows:
        p_del = r.get("P_original")
        drop = None if p_del is None or p_clean is None else p_clean - p_del
        drops.append(drop)
    usable = [(i, d) for i, d in enumerate(drops) if d is not None]
    causal = max(usable, key=lambda t: t[1])[0] if usable else None
    cited = rows[0].get("cited_index") if rows else None
    p_by_i = {r["sentence_index"]: r.get("P_original") for r in rows}
    f_sent = None
    if cited is not None and causal is not None and p_by_i.get(cited) is not None:
        others = [p for i, p in p_by_i.items() if i != cited and p is not None]
        if others:
            others_sorted = sorted(others)
            mid = others_sorted[len(others_sorted) // 2]
            f_sent = mid - p_by_i[cited]
    return {
        "id": rows[0]["id"],
        "n_sentences": len(rows),
        "causal_index": causal,
        "causal_drop": None if causal is None else drops[causal],
        "causal_sentence": None if causal is None else rows[causal].get("sentence"),
        "overlap_index": rows[0].get("overlap_index") if rows else None,
        "overlap_recall": rows[0].get("overlap_recall") if rows else None,
        "cited_index": cited,
        "cited_recall": rows[0].get("cited_recall") if rows else None,
        "cited_sentence": None if cited is None else rows[cited].get("sentence"),
        "F_sent": f_sent,
        "F_old": f_old,
        "P_clean": p_clean,
        "sentence_drops": drops,
        "flips": [r.get("flip") for r in rows],
    }
