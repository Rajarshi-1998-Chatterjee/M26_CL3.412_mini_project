"""Step 12 - Phase 4 synthesis: figures, tables, and the report skeleton."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import ensure_dir, load_json, read_jsonl, save_json


def _load(out_dir, name):
    p = Path(out_dir) / name
    if not p.exists():
        return None
    if name.endswith(".jsonl"):
        return read_jsonl(p)
    if name.endswith(".json"):
        return load_json(p)
    return None


def make_figures(out_dir, fig_dir):
    fig_dir = ensure_dir(fig_dir)
    import csv
    csv_path = Path(out_dir) / "scored_400.csv"
    scored = list(csv.DictReader(open(csv_path))) if csv_path.exists() else []
    Fs = [float(r["F"]) for r in scored if r.get("F") not in (None, "", "None")]

    # Fig 1: F(x) distribution
    if Fs:
        plt.figure(figsize=(7, 4))
        plt.hist(Fs, bins=40, color="#4C72B0", edgecolor="k", alpha=0.8)
        plt.axvline(0, color="red", ls="--", lw=1)
        plt.xlabel("F(x) = P_ctrl - P_ev")
        plt.ylabel("# questions")
        plt.title("Per-question behavioral faithfulness (N=%d)" % len(Fs))
        plt.tight_layout()
        plt.savefig(fig_dir / "fig1_F_distribution.png", dpi=150)
        plt.close()

    # Fig 2: flip rates
    p1 = _load(out_dir, "phase1_results.json")
    if p1:
        plt.figure(figsize=(5, 4))
        rates = [p1["flip_rate_evidence_removal"], p1["flip_rate_control_removal"]]
        plt.bar(["evidence-removal", "control-removal"], rates, color=["#C44E52", "#8172B3"])
        plt.ylabel("answer-flip rate")
        plt.title("Answer-flip rates under perturbation")
        for i, v in enumerate(rates):
            plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig2_flip_rates.png", dpi=150)
        plt.close()

    # Fig 3: emergence layer distribution (logit lens)
    ll = _load(out_dir, "logit_lens_100.jsonl") or []
    em = [r["emergence_layer"] for r in ll if "emergence_layer" in r]
    if em:
        plt.figure(figsize=(7, 4))
        plt.hist(em, bins=range(0, max(em) + 2), color="#55A868", edgecolor="k", alpha=0.8)
        plt.xlabel("emergence layer (first layer where chosen option leads)")
        plt.ylabel("# questions")
        plt.title("Logit-lens emergence layer (N=%d)" % len(em))
        plt.tight_layout()
        plt.savefig(fig_dir / "fig3_emergence_layer.png", dpi=150)
        plt.close()

    # Fig 4: patching recovery by layer
    pt = _load(out_dir, "patching_100.jsonl") or []
    by_layer = {}
    for r in pt:
        for p in r.get("patches", []):
            if p.get("score") is not None:
                by_layer.setdefault(p["layer"], []).append(p["score"])
    if by_layer:
        layers = sorted(by_layer)
        means = [np.mean(by_layer[l]) for l in layers]
        plt.figure(figsize=(7, 4))
        plt.plot(layers, means, marker="o", color="#CCB974")
        plt.xlabel("layer patched")
        plt.ylabel("mean recovery score")
        plt.title("Activation patching: recovery of clean answer by layer")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig4_patching_by_layer.png", dpi=150)
        plt.close()

    # Fig 5: taxonomy distribution
    tax = _load(out_dir, "taxonomy_summary.json")
    if tax and tax.get("distribution"):
        labels, counts = zip(*sorted(tax["distribution"].items(), key=lambda kv: -kv[1]))
        plt.figure(figsize=(7, 4))
        plt.bar(labels, counts, color="#64B5CD")
        plt.ylabel("# cases")
        plt.title("Error taxonomy (N=%d)" % tax["n"])
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig5_taxonomy.png", dpi=150)
        plt.close()
    return sorted(p.name for p in Path(fig_dir).glob("*.png"))


def write_report(out_dir, report_path):
    p1 = _load(out_dir, "phase1_results.json") or {}
    agr = _load(out_dir, "agreement_100.json") or {}
    tax = _load(out_dir, "taxonomy_summary.json") or {}
    stats = agr.get("stats", {})
    lines = [
        "# Faithfulness of CoT Traces in Small Medical QA Models - Results",
        "",
        "## Phase 1: Behavioral faithfulness (N=%s)" % p1.get("n"),
        f"- Clean accuracy: {p1.get('accuracy_clean')}",
        f"- Answer-flip rate, evidence-removal: {p1.get('flip_rate_evidence_removal')}",
        f"- Answer-flip rate, control-removal: {p1.get('flip_rate_control_removal')}",
        f"- Mean F(x) = P_ctrl - P_ev: {p1.get('mean_F')}",
        f"- Wilcoxon signed-rank (ev vs ctrl drops): {p1.get('wilcoxon_ev_vs_ctrl_drops')}",
        "",
        "## Phase 2: Mechanistic signals vs behavior (N=%d)" % len(agr.get("rows", [])),
    ]
    for k, v in stats.items():
        lines.append(f"- {k}: Spearman rho={v.get('spearman_rho')}, "
                     f"p={v.get('spearman_p')}, kappa={v.get('cohen_kappa')}, "
                     f"agreement={v.get('percent_agreement')}")
    lines += ["", "## Phase 3: Error taxonomy", f"- {tax}"]
    lines += ["", "## Figures", "See outputs/figures/ (fig1..fig5).", ""]
    path = ensure_dir(Path(report_path).parent) / Path(report_path).name
    path.write_text("\n".join(lines))
    return path


def run_report(out_dir, fig_dir=None):
    out_dir = Path(out_dir)
    fig_dir = fig_dir or out_dir / "figures"
    figs = make_figures(out_dir, fig_dir)
    report = write_report(out_dir, out_dir / "report.md")
    return {"figures": figs, "report": str(report)}
