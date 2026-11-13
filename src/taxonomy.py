"""Steps 8/11 - Phase 3 error taxonomy (human annotation + agreement).

Workflow (needs HUMAN input):
  1. export_annotation_sheet(...) writes a CSV of the N=50 review cases
     (id, question, options, trace, predicted vs gold, F(x), disagreement flags).
  2. BOTH annotators independently fill in `primary_label` (and optional
     `secondary_label`) and return labels_annotator1.csv / labels_annotator2.csv.
  3. compute_agreement(...) reports raw % agreement + Cohen's kappa
     (Landis & Koch: kappa > 0.6 = "substantial").
  4. Disputed cases are adjudicated by discussion -> final labels CSV.

One PRIMARY label per case keeps kappa well-defined; secondary labels are
reported descriptively only.
"""
import csv
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from .common import ensure_dir, read_jsonl, save_json

CATEGORIES = [
    "knowledge_gap",                # model lacks the medical fact
    "reasoning_slip",               # logic error given correct facts
    "hallucinated_premise",         # invents a finding not in the vignette
    "question_misreading",           # misreads the question/options
    "confident_unjustified_leap",   # correct answer, no valid justification
    "other",
]


def select_review_cases(scored, off_diagonal_ids, n_total=50, traces=None):
    """Pick N=50 review cases: all off-diagonal disagreement cases first, then
    fill with incorrect or lowest-F cases (low-faithfulness priority)."""
    off = set(off_diagonal_ids or [])
    scored_sorted = sorted(scored, key=lambda r: (r.get("F") if r.get("F") is not None else 0.0))
    chosen_ids = []
    for r in scored:
        if r["id"] in off:
            chosen_ids.append(r["id"])
    for r in scored_sorted:
        if len(chosen_ids) >= n_total:
            break
        if r["id"] not in chosen_ids and (not r.get("correct") or (r.get("F") is not None and r["F"] <= 0)):
            chosen_ids.append(r["id"])
    # top up with lowest-F regardless of correctness
    for r in scored_sorted:
        if len(chosen_ids) >= n_total:
            break
        if r["id"] not in chosen_ids:
            chosen_ids.append(r["id"])
    return chosen_ids[:n_total]


def export_annotation_sheet(case_ids, items, traces, scored, out_csv):
    """Write the annotation sheet CSV for human review."""
    out_csv = ensure_dir(Path(out_csv).parent) / Path(out_csv).name
    items_d = {r["id"]: r for r in items}
    tr_d = {r["id"]: r for r in traces or []}
    sc_d = {r["id"]: r for r in scored}
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "options", "trace", "predicted", "gold",
                    "correct", "F", "primary_label", "secondary_label", "notes"])
        for cid in case_ids:
            it, tr, sc = items_d.get(cid, {}), tr_d.get(cid, {}), sc_d.get(cid, {})
            w.writerow([
                cid,
                it.get("question", ""),
                " | ".join(f"{k}) {v}" for k, v in sorted(it.get("options", {}).items())),
                (tr.get("trace", "") or "")[:4000],
                tr.get("predicted", ""), it.get("gold", ""),
                sc.get("correct", ""), sc.get("F", ""),
                "", "", "",  # annotator fills these
            ])
    return out_csv


def load_labels(path):
    """Load an annotator CSV -> dict id -> {primary, secondary}."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["id"]] = {
                "primary": (row.get("primary_label") or "").strip(),
                "secondary": (row.get("secondary_label") or "").strip(),
            }
    return out


def compute_agreement(labels1, labels2):
    """Raw % agreement + Cohen's kappa over shared ids (primary labels)."""
    shared = sorted(set(labels1) & set(labels2))
    if not shared:
        return {"n": 0, "note": "no shared ids"}
    a = [labels1[i]["primary"] for i in shared]
    b = [labels2[i]["primary"] for i in shared]
    agree = sum(1 for x, y in zip(a, b) if x == y) / len(shared)
    kappa = cohen_kappa_score(a, b) if len(set(a) | set(b)) > 1 else None
    disputed = [i for i, (x, y) in zip(shared, zip(a, b)) if x != y]
    return {
        "n": len(shared),
        "raw_agreement": float(agree),
        "cohen_kappa": float(kappa) if kappa is not None else None,
        "landis_koch": ("substantial" if kappa and kappa > 0.6 else
                        ("moderate" if kappa and kappa > 0.4 else "lower")),
        "disputed_ids": disputed,
    }


def finalize_labels(labels1, labels2, adjudicated, out_csv):
    """Write final adjudicated label file. `adjudicated`: dict id -> primary label
    for disputed cases (human decision)."""
    out_csv = ensure_dir(Path(out_csv).parent) / Path(out_csv).name
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "primary_label", "secondary_label", "source"])
        for cid in sorted(set(labels1) | set(labels2)):
            l1 = labels1.get(cid, {}).get("primary", "")
            l2 = labels2.get(cid, {}).get("primary", "")
            if cid in adjudicated:
                prim, src = adjudicated[cid], "adjudicated"
            elif l1 == l2:
                prim, src = l1, "agreed"
            else:
                prim, src = l1, "unresolved"
            sec = labels1.get(cid, {}).get("secondary", "")
            w.writerow([cid, prim, sec, src])
    return out_csv


def taxonomy_summary(final_csv, out_json=None):
    """Label distribution over the final adjudicated set."""
    from collections import Counter
    with open(final_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    dist = Counter(r["primary_label"] for r in rows)
    res = {"n": len(rows), "distribution": dict(dist),
           "unresolved": sum(1 for r in rows if r["source"] == "unresolved")}
    if out_json:
        save_json(res, out_json)
    return res
