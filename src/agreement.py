"""Step 10 - Agreement between behavioral (Phase 1) and mechanistic (Phase 2)
signals on the N=100 subset.

- Spearman rank correlation rho between F(x) and each mechanistic score
  (logit-lens earliness, patching best recovery) - rank-based because N is
  small and the relationship may be non-linear/non-normal.
- Binarized agreement: threshold both scores (median split, or F(x) > 0),
  report raw % agreement + Cohen's kappa, and the 2x2 contingency table.
- Off-diagonal cases (the two methods disagree) are routed to the Phase 3
  taxonomy review.
"""
import json

from scipy import stats
from sklearn.metrics import cohen_kappa_score

from .common import save_json


def _median_split(values):
    s = sorted(v for v in values if v is not None)
    if not s:
        return 0.0
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def join_signals(scored, logit_lens_recs, patching_recs):
    """Join per-question F(x) with mechanistic scores by id."""
    ll = {r["id"]: r for r in logit_lens_recs if "earliness" in r}
    pt = {r["id"]: r for r in patching_recs if r.get("best")}
    rows = []
    for r in scored:
        if r.get("F") is None:
            continue
        row = {"id": r["id"], "F": r["F"], "correct": r.get("correct"),
               "flip_ev": r.get("flip_ev")}
        if r["id"] in ll:
            row["earliness"] = ll[r["id"]]["earliness"]
            row["emergence_layer"] = ll[r["id"]]["emergence_layer"]
        if r["id"] in pt:
            row["patch_best"] = pt[r["id"]]["best"]["score"]
            row["patch_best_layer"] = pt[r["id"]]["best"]["layer"]
        rows.append(row)
    return rows


def agreement_stats(rows, f_threshold=None):
    """Spearman rho + binarized agreement (Cohen's kappa) for each mechanistic
    score against F(x). Returns dict of stats and the off-diagonal ids."""
    out = {}
    mech_keys = [k for k in ("earliness", "patch_best") ]
    for key in mech_keys:
        pairs = [(r["F"], r[key]) for r in rows if r.get(key) is not None]
        if len(pairs) < 10:
            out[key] = {"n": len(pairs), "note": "too few paired points"}
            continue
        fs = [p[0] for p in pairs]
        ms = [p[1] for p in pairs]
        rho, pval = stats.spearmanr(fs, ms)
        f_thr = f_threshold if f_threshold is not None else _median_split(fs)
        m_thr = _median_split(ms)
        f_bin = [1 if f > f_thr else 0 for f in fs]
        m_bin = [1 if m > m_thr else 0 for m in ms]
        agree = sum(1 for a, b in zip(f_bin, m_bin) if a == b) / len(f_bin)
        kappa = cohen_kappa_score(f_bin, m_bin)
        # 2x2 table: rows behavioral, cols mechanistic
        tp = sum(1 for a, b in zip(f_bin, m_bin) if a == 1 and b == 1)
        tn = sum(1 for a, b in zip(f_bin, m_bin) if a == 0 and b == 0)
        fp = sum(1 for a, b in zip(f_bin, m_bin) if a == 1 and b == 0)  # behav faithful, mech flagged
        fn = sum(1 for a, b in zip(f_bin, m_bin) if a == 0 and b == 1)
        off_diag = [r["id"] for r, a, b in zip(rows, f_bin, m_bin)
                    if r.get(key) is not None and a != b]
        out[key] = {
            "n": len(pairs),
            "spearman_rho": float(rho),
            "spearman_p": float(pval),
            "f_threshold": f_thr,
            "mech_threshold": m_thr,
            "percent_agreement": float(agree),
            "cohen_kappa": float(kappa),
            "contingency_2x2": {"both_faithful": tp, "both_unfaithful": tn,
                                "behav_faithful_mech_flagged": fp,
                                "behav_unfaithful_mech_faithful": fn},
            "off_diagonal_ids": off_diag,
        }
    return out


def run_agreement(scored, logit_lens_recs, patching_recs, out_dir, f_threshold=None):
    from .common import ensure_dir
    out_dir = ensure_dir(out_dir)
    rows = join_signals(scored, logit_lens_recs, patching_recs)
    stats_res = agreement_stats(rows, f_threshold=f_threshold)
    save_json({"rows": rows, "stats": stats_res}, out_dir / "agreement_100.json")
    # union of off-diagonal ids across mechanistic scores -> taxonomy review pool
    off = set()
    for v in stats_res.values():
        off.update(v.get("off_diagonal_ids", []))
    save_json(sorted(off), out_dir / "off_diagonal_ids.json")
    return stats_res, rows
