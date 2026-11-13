#!/usr/bin/env python3
"""Generate Colab-ready notebooks (00-05) as valid .ipynb JSON (stdlib only)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks"
NB.mkdir(exist_ok=True)

SETUP_CELL = '''\
# @title Setup (run once per session)
import os, sys
ROOT = "/content/mini_project"
if not os.path.isdir(ROOT):
    raise SystemExit(
        "Upload the mini_project folder to Colab first: Files panel -> Upload -> "
        "mini_project.zip, then unzip: !unzip -q mini_project.zip -d /content/  "
        "(or git clone your repo into /content/mini_project)"
    )
os.chdir(ROOT)
sys.path.insert(0, ROOT)  # project root, so `src` imports as a package (src/*.py use relative imports)
get_ipython().system("pip install -q -r requirements.txt")
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# <-- SET ME: path to your `data` folder in Google Drive, e.g.
#     "/content/drive/MyDrive/mini_project/data". Leave "" to use the local
#     ./data folder inside this repo instead (no Drive mount).
DATA_DIR = ""
if DATA_DIR:
    from google.colab import drive
    drive.mount("/content/drive")
    os.environ["MINI_PROJECT_DATA_DIR"] = DATA_DIR
    print("data dir ->", DATA_DIR)
# HF token: only needed for GATED models (not for the primary Qwen medical model)
import getpass
_hf = getpass.getpass("HF token (press Enter to skip): ")
if _hf:
    os.environ["HF_TOKEN"] = _hf
print("setup OK")
'''

DOWNLOAD_CELL = '''\
# @title Download outputs (run after the phase finishes)
import os, zipfile, glob
from google.colab import files
out_dir = "outputs"
zip_path = "/content/outputs.zip"
with zipfile.ZipFile(zip_path, "w") as z:
    for f in glob.glob(os.path.join(out_dir, "**", "*"), recursive=True):
        if os.path.isfile(f):
            z.write(f, os.path.relpath(f, "/content"))
files.download(zip_path)
print("Downloaded", zip_path)
'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "cells": cells,
    }


notebooks = {
    "00_setup_and_smoke.ipynb": [
        md("# 00 - Setup & smoke test\\n\\nVerifies GPU, installs deps, and runs a 3-question "
           "smoke generation with the primary model. **Expected:** 3 traces with parsed answers."),
        code(SETUP_CELL),
        code('''\
from src.common import load_config, load_prompt_config, load_model_and_tokenizer, set_seed
from src.inference import build_prompt, generate_and_score, parse_answer_letter, letter_token_ids
cfg = load_config()
pcfg = load_prompt_config()
set_seed(cfg["seed"])
model_id = cfg["model"]["primary"]
print("model:", model_id)
model, tok = load_model_and_tokenizer(model_id, dtype=cfg["model"]["dtype"])
# tiny smoke set: first 3 MedQA questions
from src.data_loading import load_medqa
items = load_medqa(n=3, seed=cfg["seed"])
for it in items:
    rec = generate_and_score(model, tok, pcfg, it, max_new_tokens=256)
    print("="*80)
    print("Q:", it["question"][:200])
    print("gold:", it["gold"], "| predicted:", rec["predicted"], "| P_clean:", rec["P_clean"])
    print("trace tail:", rec["trace"][-300:])
'''),
    ],
    "01_phase1_behavioral.ipynb": [
        md("# 01 - Phase 1: behavioral faithfulness (Week 1)\\n\\nSteps 2-5: sample 400 MedQA "
           "questions, generate CoT traces, build evidence/control perturbations, score them, "
           "compute F(x) + flip rates + Wilcoxon, and select the 100-question mechanistic subset.\\n\\n"
           "**Runtime on a T4 GPU:** a few hours for 400 questions x 3 runs. Results checkpoint after "
           "every question, so you can stop/resume safely (re-running skips finished ids)."),
        code(SETUP_CELL),
        code('''\
from src import pipeline
from src.common import load_config
cfg = load_config()
# Override the model here if you want a different candidate:
# cfg["model"]["primary"] = "akshayballal/Qwen3-1.7B-Pubmed-16bit-GRPO"
stats = pipeline.run_phase1(cfg)
print("PHASE 1 DONE")
import json; print(json.dumps(stats, indent=2))
'''),
        code(DOWNLOAD_CELL),
    ],
    "02_phase2_logit_lens.ipynb": [
        md("# 02 - Phase 2a: logit lens on the 100-question subset (Week 2)\\n\\nStep 7: per-layer "
           "probability of each option letter at the answer position; emergence layer + earliness "
           "score. Requires Phase 1 outputs (run notebook 01 first, or place its outputs in `outputs/`)."),
        code(SETUP_CELL),
        code('''\
from src import pipeline
from src.common import load_config
cfg = load_config()
recs = pipeline.run_phase2_logit_lens(cfg)
ok = [r for r in recs if "earliness" in r]
print(f"logit lens done for {len(ok)} questions")
if ok:
    import statistics
    print("mean emergence layer:", statistics.mean(r["emergence_layer"] for r in ok))
    print("mean earliness:", statistics.mean(r["earliness"] for r in ok))
'''),
        code(DOWNLOAD_CELL),
    ],
    "03_phase2_patching_and_agreement.ipynb": [
        md("# 03 - Phase 2b: activation patching + behavioral/mechanistic agreement (Week 3)\\n\\n"
           "Step 9: patch clean activations into the corrupted run at each (layer, position) of the "
           "redacted span; recovery score on the correct option's logit. "
           "Step 10: Spearman rho + Cohen's kappa between F(x) and mechanistic scores; "
           "off-diagonal cases are routed to the Phase 3 review."),
        code(SETUP_CELL),
        code('''\
from src import pipeline
from src.common import load_config
cfg = load_config()
recs = pipeline.run_phase2_patching(cfg)
ok = [r for r in recs if r.get("best")]
print(f"patching done for {len(ok)} questions")
stats, case_ids = pipeline.run_agreement(cfg)
import json; print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "off_diagonal_ids"}
      for k, v in stats.items()}, indent=2))
print("review case ids:", len(case_ids))
'''),
        code(DOWNLOAD_CELL),
    ],
    "04_phase3_taxonomy.ipynb": [
        md("# 04 - Phase 3: error taxonomy annotation (Weeks 2-3, NEEDS HUMAN INPUT)\\n\\n"
           "Step 8/11. Run the first cell to export `outputs/annotation_sheet_50.csv` "
           "(N=50 review cases: disagreement cases + incorrect/low-F).\\n\\n"
           "**Human task:** BOTH annotators independently fill the `primary_label` and optional "
           "`secondary_label` columns (categories: knowledge_gap, reasoning_slip, "
           "hallucinated_premise, question_misreading, confident_unjustified_leap, other) and save "
           "as `labels_annotator1.csv` and `labels_annotator2.csv`. Upload both back into "
           "`outputs/` and run the last cell to compute agreement (raw % + Cohen's kappa) "
           "and the final adjudicated labels."),
        code(SETUP_CELL),
        code('''\
import json
from src.common import load_config, read_jsonl, read_records, save_json
from src.taxonomy import export_annotation_sheet, select_review_cases
cfg = load_config()
n = cfg["dataset"]["n_sample"]
data_dir = cfg["paths"]["data_dir"]
out_dir = "outputs"
scored = read_records(f"{out_dir}/scored_{n}.json")
items = read_jsonl(f"{data_dir}/subset_100.jsonl")
traces = read_jsonl(f"{out_dir}/traces_{n}.jsonl")
# full 400-item pool for review cases
all_items = read_jsonl(f"{data_dir}/sample_{n}.jsonl")
off = json.load(open(f"{out_dir}/off_diagonal_ids.json")) if __import__("os").path.exists(f"{out_dir}/off_diagonal_ids.json") else []
case_ids = select_review_cases(scored, off, n_total=50, traces=traces)
save_json(case_ids, f"{out_dir}/review_case_ids.json")
csv_path = export_annotation_sheet(case_ids, all_items, traces, scored,
                                   f"{out_dir}/annotation_sheet_50.csv")
print("Wrote", csv_path, "with", len(case_ids), "cases - download it and annotate.")
'''),
        code('''\
# After uploading labels_annotator1.csv and labels_annotator2.csv into outputs/:
from src.taxonomy import load_labels, compute_agreement, finalize_labels, taxonomy_summary
l1 = load_labels("outputs/labels_annotator1.csv")
l2 = load_labels("outputs/labels_annotator2.csv")
agree = compute_agreement(l1, l2)
print("Inter-annotator agreement:", agree)
# Adjudication: fill this dict with your discussed decision for each disputed id
adjudicated = {cid: "" for cid in agree["disputed_ids"]}  # <-- HUMAN INPUT
final_csv = finalize_labels(l1, l2, adjudicated, "outputs/taxonomy_labels_final.csv")
summary = taxonomy_summary(final_csv, out_json="outputs/taxonomy_summary.json")
print("Taxonomy summary:", summary)
'''),
        code(DOWNLOAD_CELL),
    ],
    "05_phase4_report.ipynb": [
        md("# 05 - Phase 4: synthesis & report (Week 4)\\n\\nStep 12: figures + `outputs/report.md` "
           "from all phase outputs. Also zips everything for download."),
        code(SETUP_CELL),
        code('''\
from src import pipeline
from src.common import load_config
cfg = load_config()
res = pipeline.run_report(cfg)
print("figures:", res["figures"])
print("report:", res["report"])
print(open(res["report"]).read()[:2000])
'''),
        code(DOWNLOAD_CELL),
    ],
}

for name, cells in notebooks.items():
    with open(NB / name, "w") as f:
        json.dump(nb(cells), f, indent=1)
    print("wrote", NB / name)
