# Faithfulness of CoT Traces in Small Medical QA Models — Implementation

Code + plan for the mini-project described in `MI_mini_project_proposal.pdf`
(see `PLAN.md` for the step-by-step plan this implements).

## Layout
```
mini_project/
├── PLAN.md                  # step-by-step plan
├── README.md                # this file
├── requirements.txt
├── configs/default.yaml     # model ids, sample sizes, seed, paths
├── configs/prompt.yaml      # shared few-shot CoT prompt template
├── eval_faithfulness.py     # reusable CLI (Phase 1 only, no mechanistic deps)
├── src/
│   ├── common.py            # config/IO/seeds/model loading/answer parsing
│   ├── data_loading.py      # MedQA loading + fixed-seed sampling + subset selection
│   ├── inference.py         # CoT generation + answer-probability scoring (resumable)
│   ├── perturbations.py     # evidence-removal vs control-removal vignettes
│   ├── faithfulness.py      # F(x), flip rates, Wilcoxon test
│   ├── logit_lens.py        # Phase 2a logit lens (TransformerLens)
│   ├── patching.py          # Phase 2b activation patching + recovery score
│   ├── agreement.py         # Spearman rho, Cohen's kappa, 2x2 table
│   ├── taxonomy.py          # Phase 3 annotation sheet + inter-annotator agreement
│   ├── report.py            # Phase 4 figures + report.md
│   └── pipeline.py          # phase orchestrators used by the notebooks/CLI
├── scripts/make_notebooks.py
└── notebooks/               # Colab notebooks 00-05 (generated)
```

## Running on Colab (GPU) — the intended workflow
1. Zip this folder and upload to Colab (Files panel → Upload `mini_project.zip`),
   then in a notebook cell: `!unzip -q mini_project.zip -d /content/` — or `git clone` it.
2. Runtime → Change runtime type → GPU (T4 is enough for 1–3B models).
3. **Use the combined notebook `notebooks/faithfulness_project.ipynb`** (all phases
   00–05 in one file, run top to bottom) — or run the individual notebooks
   `00_setup_and_smoke` → `01_phase1_behavioral` → `02_phase2_logit_lens` →
   `03_phase2_patching_and_agreement` → `04_phase3_taxonomy` → `05_phase4_report`
   in order. Phase 1 takes a few hours on T4 but is resumable.
5. Each notebook ends with a cell that zips `outputs/` and downloads it.
   **Send me the outputs zip after each phase** and I'll continue the next step.

## Human input required 
- **HF token** — only if you want a *gated* candidate (Llama-3.2-1B-based). The
  primary model (`Tianye88/Qwen2.5-1.5B-Instruct-Medical-cpt-sft-v1`) is ungated.
- **Phase 3 taxonomy labels** — both team members annotate the 50-case sheet
  (`outputs/annotation_sheet_50.csv`) independently; upload the two label CSVs back.
- **Adjudication** of disputed taxonomy labels.

## Local / CLI usage (no GPU needed for code inspection; CPU works for smoke tests)
```bash
pip install -r requirements.txt
# smoke test (20 questions, any model id):
python eval_faithfulness.py --smoke-test --model <hf-model-id> --out outputs/smoke/
# full Phase 1:
python eval_faithfulness.py --model Tianye88/Qwen2.5-1.5B-Instruct-Medical-cpt-sft-v1 \
    --dataset medqa --n 400 --out outputs/run1/
# local MedQA-style JSONL ({id, question, options, gold}):
python eval_faithfulness.py --dataset local --dataset-path data/my.jsonl --n 100
```

## Outputs produced (share these back after each Colab run)
| File | Produced by | Contents |
|---|---|---|
| `data/sample_400.jsonl` | 01 | fixed-seed MedQA sample |
| `outputs/traces_400.jsonl` | 01 | CoT traces, parsed answers, P_clean |
| `outputs/perturbations_400.jsonl` | 01 | evidence/control redacted vignettes |
| `outputs/ev_scores_400.jsonl`, `ctrl_scores_400.jsonl` | 01 | P(original answer), flip |
| `outputs/phase1_results.json`, `scored_400.csv/json` | 01 | F(x), flip rates, Wilcoxon |
| `data/subset_100_ids.json`, `subset_100.jsonl` | 01 | mechanistic subset |
| `outputs/logit_lens_100.jsonl` | 02 | emergence layer, earliness |
| `outputs/patching_100.jsonl` | 03 | recovery scores per (layer,pos) |
| `outputs/agreement_100.json`, `off_diagonal_ids.json` | 03 | rho, kappa, 2x2 |
| `outputs/annotation_sheet_50.csv`, `taxonomy_*` | 04 | human labels + agreement |
| `outputs/report.md`, `outputs/figures/*.png` | 05 | final report |
