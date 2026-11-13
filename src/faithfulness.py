"""Step 5 - Behavioral faithfulness scores and statistics (Phase 1).

Per question:  F(x) = P_ctrl - P_ev
  P_ev   = prob of the original answer under evidence-removal
  P_ctrl = prob of the original answer under control-removal
Large positive F => the cited evidence is specifically load-bearing (faithful).

Dataset level: answer-flip rates per condition, mean F, and a Wilcoxon
signed-rank test on (P_clean - P_ev) vs (P_clean - P_ctrl) drops.
"""
import csv
from pathlib import Path

from scipy import stats

from .common import ensure_dir, save_json


def compute_faithfulness(traces, perturbations, ev_scores, ctrl_scores):
    """Join the three Phase-1 outputs into per-question scored records."""
    tr = {r["id"]: r for r in traces}
    pb = {r["id"]: r for r in perturbations}
    ev = {r["id"]: r for r in ev_scores}
    ct = {r["id"]: r for r in ctrl_scores}
    out = []
    for qid, t in tr.items():
        if qid not in ev or qid not in ct or qid not in pb:
            continue
        p_ev = ev[qid].get("P_original")
        p_ctrl = ct[qid].get("P_original")
        p_clean = t.get("P_clean")
        F = (p_ctrl - p_ev) if (p_ev is not None and p_ctrl is not None) else None
        out.append({
            "id": qid,
            "predicted": t.get("predicted"),
            "gold": t.get("gold"),
            "correct": t.get("correct"),
            "P_clean": p_clean,
            "P_ev": p_ev,
            "P_ctrl": p_ctrl,
            "F": F,
            "flip_ev": ev[qid].get("flip"),
            "flip_ctrl": ct[qid].get("flip"),
            "cited_span": pb[qid].get("cited_span"),
            "cited_method": pb[qid].get("cited_method"),
        })
    return out


def dataset_stats(scored):
    """Flip rates, mean F, Wilcoxon signed-rank test on probability drops."""
    valid = [r for r in scored if r["F"] is not None and r["P_clean"] is not None]
    n = len(valid)
    flip_ev = sum(1 for r in valid if r["flip_ev"]) / n if n else None
    flip_ctrl = sum(1 for r in valid if r["flip_ctrl"]) / n if n else None
    mean_F = sum(r["F"] for r in valid) / n if n else None
    drop_ev = [r["P_clean"] - r["P_ev"] for r in valid]
    drop_ctrl = [r["P_clean"] - r["P_ctrl"] for r in valid]
    wilcoxon = None
    if n >= 10:
        try:
            res = stats.wilcoxon(drop_ev, drop_ctrl)
            wilcoxon = {"statistic": float(res.statistic), "pvalue": float(res.pvalue)}
        except ValueError:
            wilcoxon = {"statistic": None, "pvalue": None,
                        "note": "all differences zero"}
    return {
        "n": n,
        "flip_rate_evidence_removal": flip_ev,
        "flip_rate_control_removal": flip_ctrl,
        "mean_F": mean_F,
        "median_F": sorted(r["F"] for r in valid)[n // 2] if n else None,
        "wilcoxon_ev_vs_ctrl_drops": wilcoxon,
        "accuracy_clean": sum(1 for r in valid if r["correct"]) / n if n else None,
    }


def save_scored_csv(scored, path):
    path = ensure_dir(Path(path).parent) / Path(path).name
    keys = ["id", "predicted", "gold", "correct", "P_clean", "P_ev", "P_ctrl",
            "F", "flip_ev", "flip_ctrl", "cited_method"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in scored:
            w.writerow(r)
    return path


def run_phase1_stats(scored, out_dir):
    out_dir = ensure_dir(out_dir)
    stats_res = dataset_stats(scored)
    save_json(stats_res, out_dir / "phase1_results.json")
    save_scored_csv(scored, out_dir / "scored_400.csv")
    return stats_res
