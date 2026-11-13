#!/usr/bin/env python3
"""Join notebooks/00-05 into one combined notebook (stdlib only)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks"

ORDER = [
    ("00_setup_and_smoke.ipynb", "Phase 0 - Setup & smoke test",
     ["Step 0 - Environment & project setup (install deps, GPU check, HF token)",
      "Step 1 - Model vetting: smoke-test generation with the primary model"]),
    ("01_phase1_behavioral.ipynb", "Phase 1 - Behavioral faithfulness (Week 1)",
     ["Step 0 - Environment & project setup (install deps, GPU check, HF token)",
      "Steps 2-5 - Phase 1: MedQA sample -> CoT generation (Step 3) -> evidence/control perturbations (Step 4) -> F(x) + flip rates + Wilcoxon (Step 5) -> 100-question subset",
      "Download outputs.zip (share with me after Phase 1)"]),
    ("02_phase2_logit_lens.ipynb", "Phase 2a - Logit lens (Week 2)",
     ["Step 0 - Environment & project setup",
      "Step 7 - Phase 2a: logit lens on the 100-question subset (emergence layer, earliness)",
      "Download outputs.zip"]),
    ("03_phase2_patching_and_agreement.ipynb", "Phase 2b - Activation patching + agreement (Week 3)",
     ["Step 0 - Environment & project setup",
      "Steps 9-10 - Phase 2b: activation patching (Step 9) + behavioral/mechanistic agreement: Spearman rho, Cohen's kappa, off-diagonal routing (Step 10)",
      "Download outputs.zip"]),
    ("04_phase3_taxonomy.ipynb", "Phase 3 - Error taxonomy annotation (needs human input)",
     ["Step 0 - Environment & project setup",
      "Step 8 - Phase 3: export annotation_sheet_50.csv (N=50 review cases) - HUMAN ANNOTATION FOLLOWS",
      "Step 11 - Phase 3: inter-annotator agreement (raw % + Cohen's kappa) + adjudicated final labels - needs the two label CSVs uploaded",
      "Download outputs.zip"]),
    ("05_phase4_report.ipynb", "Phase 4 - Synthesis & report (Week 4)",
     ["Step 0 - Environment & project setup",
      "Step 12 - Phase 4: consolidate results, figures, report.md (Steps 12-14 deliverables)",
      "Download outputs.zip (final)"]),
]


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


cells = [md(
    "# Faithfulness of CoT Traces in Small Medical QA Models - Full Pipeline\n\n"
    "Combined notebook (phases 00-05 in order). Run top to bottom on a Colab GPU runtime.\n"
    "Upload `mini_project.zip` to Colab and `!unzip -q mini_project.zip -d /content/` first; "
    "the first setup cell verifies this. After each phase, run that phase's download cell "
    "and share `outputs.zip`. Phase 3 (taxonomy) requires human annotation - see its markdown cell."
)]
for fname, title, code_labels in ORDER:
    nb = json.load(open(NB / fname))
    cells.append(md(f"\n---\n\n# {title}\n\n*(from `{fname}`)*"))
    li = 0
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            label = code_labels[li] if li < len(code_labels) else f"Code cell {li+1}"
            li += 1
            cells.append(md(f"**▶ {label}**"))
        cells.append(cell)

out = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                 "language_info": {"name": "python"}},
    "cells": cells,
}
dst = NB / "faithfulness_project.ipynb"
json.dump(out, open(dst, "w"), indent=1)
print("wrote", dst, "with", len(cells), "cells")
