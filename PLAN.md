# Plan: Probing the Faithfulness of Reasoning Traces in Small LMs for Medical QA

**Source document:** `MI_mini_project_proposal.pdf`
**Goal:** Determine whether chain-of-thought (CoT) traces from small (1–3B) open-weight medical QA models on MedQA *faithfully* reflect the computation that produces the answer — combining behavioral perturbation tests with mechanistic (white-box) analysis, plus a qualitative error taxonomy.
**Timeline:** 4 weeks. **Team:** 2 people. **Hardware:** one consumer/cloud-grade GPU is enough.

---

## Step 0 — Environment & Project Setup (Day 1)

**Requirements:** Python 3.10+, one GPU (e.g., RTX 3060/4060 or better; 1–3B models fit in ~8–12 GB VRAM in bf16).

**Coding tasks — in detail:**
1. Create a project repo structure:
   ```
   mini_project/
   ├── PLAN.md                 # this file
   ├── requirements.txt
   ├── configs/                # YAML/JSON configs: model name, sample sizes, seeds
   ├── data/                   # MedQA sample (git-ignored if large)
   ├── src/
   │   ├── data_loading.py     # MedQA download + sampling
   │   ├── inference.py        # CoT generation wrapper
   │   ├── perturbations.py    # evidence/control removal
   │   ├── faithfulness.py     # F(x), flip rates, Wilcoxon test
   │   ├── logit_lens.py       # mechanistic probe
   │   ├── patching.py         # activation patching
   │   ├── taxonomy.py         # annotation + agreement stats
   │   └── report.py           # tables/figures for write-up
   ├── notebooks/              # exploratory analysis
   └── outputs/                # results JSON/CSV, figures (git-ignored)
   ```
2. `requirements.txt` must include at minimum: `torch`, `transformers`, `datasets` (Hugging Face, for MedQA), `accelerate`, `scipy` (Wilcoxon, Spearman), `scikit-learn` (Cohen's κ), `matplotlib`/`seaborn` (figures), and either `transformer_lens` **or** `nnsight` for mechanistic work. Pin versions for reproducibility.
3. Set a global seed everywhere (`torch.manual_seed`, `numpy`, `random`) and log the config used for every run so results are reproducible.
4. Get Hugging Face access set up (some models are gated — request access early) and verify you can load and run a 1–3B model end-to-end with a smoke-test prompt before Week 1 experiments.

---

## Step 1 — Model Selection & Vetting (Day 1–2)

**Requirement:** Pick **one primary model** and keep the others as secondary/robustness checks. Candidates from the proposal (all instruction-tuned, open-weight, 1–3B, medically adapted):

| Candidate | Base | Notes |
|---|---|---|
| Qwen2.5-1.5B-Instruct-Medical | Qwen2.5-1.5B-Instruct | **Primary candidate** (well-documented base) |
| Bio-Medical-Llama-3.2-1B-CoT | Llama-3.2-1B-Instruct | Fine-tuned for biomedical CoT — directly on-topic |
| Qwen3-1.7B-Pubmed-GRPO | Qwen3-1.7B | Secondary cross-check (different recipe) |
| TinyLlama-1.1B-Medical | TinyLlama-1.1B | Lower-capacity check |

**Coding tasks — in detail:**
1. Write `src/model_selection.py` (or a notebook) that loads each candidate via `transformers.AutoModelForCausalLM.from_pretrained(..., torch_dtype=bfloat16, device_map="auto")` and runs a fixed 20-question smoke set.
2. Record for each: does it follow few-shot CoT format reliably, does it emit a parseable final answer (e.g., `Answer: X`), latency per question, VRAM usage. Check model cards for evaluation numbers before final selection.
3. Implement a **robust answer parser** (regex over the trace tail, e.g. `Answer:\s*([A-E])`) — every downstream score depends on extracting the option letter cleanly. Log unparseable outputs explicitly.
4. Decide and freeze the **few-shot CoT prompt template** (2–3 medical QA examples with reasoning + `Answer: X`), stored in `configs/prompt.yaml` so all experiments share it.

---

## Step 2 — Dataset: MedQA Sample (Day 2)

**Requirement:** Primary dataset is **MedQA** (USMLE-style, 4/5-option MCQs with clean gold labels — clean labels make causal interventions trivially scoreable). **PubMedQA** is a lower-priority secondary robustness check (yes/no/maybe — noisier; stretch goal only).

**Coding tasks — in detail:**
1. `src/data_loading.py`: load MedQA via Hugging Face `datasets` (e.g., `bigbio/med_qa` or the original `jsonl` release), keep fields: `question`, `options` (dict of letter → text), `answer_idx`/gold letter, and the vignette/context.
2. Draw a **fixed random sample of N = 400 questions** (`random.Random(42).sample(...)`), stratified if feasible so both 4- and 5-option questions are represented; save the sampled IDs to `data/sample_400_ids.json` so Phases 1–3 all use the *same* questions.
3. From the 400, pre-select the **targeted subset of N = 100** for mechanistic work: drawn from the Phase 1 sample, **oversampling low-faithfulness cases** (do this *after* Phase 1 scoring — so Step 2b runs at the start of Week 2). Save to `data/subset_100_ids.json`.
4. Persist the sample as CSV/JSONL with columns: `id, question, options, gold_answer`.

---

## Week 1 — Phase 1: Behavioral Faithfulness via Causal Ablation (Steps 3–5)

### Step 3 — CoT Trace Generation (N = 400)
**Coding tasks — in detail:**
1. `src/inference.py`: a function `generate_cot(model, tokenizer, prompt) -> {trace, answer_letter, answer_logprob}` that (a) builds the few-shot prompt, (b) generates with `model.generate(..., max_new_tokens=512, do_sample=False)` (greedy decoding for reproducibility), (c) parses the final option letter, and (d) computes the **probability the model assigns to its own chosen answer** — use the token log-probs of the answer letter token(s) from the returned `scores`, or re-run a forward pass over the trace and read the logit at the answer position. Save per-question: `trace_text`, `predicted_letter`, `P_clean` (probability of the predicted answer under the clean prompt).
2. Batch the 400 questions (batch size limited by VRAM), checkpoint outputs to `outputs/traces_400.jsonl` incrementally so a crash doesn't lose work.

### Step 4 — Counterfactual Perturbations (evidence-removal vs control-removal)
**Requirement:** For each question, build **two** modified vignettes (option list unchanged in both):
- **Evidence-removal**: delete/alter the *specific span the CoT trace cites as decisive* (e.g., the symptom/lab value the trace says determined the answer).
- **Control-removal**: delete/alter a *different, uncited span of comparable length*.

The control is what distinguishes "sensitive to losing the cited evidence" from "sensitive to losing any information."

**Coding tasks — in detail:**
1. `src/perturbations.py`:
   - Implement span identification: first try an automated approach — ask the model (or match heuristically) which vignette span the trace cites; fall back to manual selection for ambiguous cases. Store `cited_span` per question.
   - `make_evidence_removal(question, cited_span)` → vignette with that span deleted/replaced (e.g., "[REDACTED]").
   - `make_control_removal(question, cited_span)` → pick a random *uncited* span of similar token length (seeded), delete/replace it.
   - Sanity checks: option list byte-identical across clean/evidence/control; perturbed vignette length within ±20% of original.
2. Re-run **only the answer-selection step** on each perturbed prompt (cheap: no need to regenerate the full trace — but keep the trace prefix fixed if simpler; document the choice) and record `P_ev` and `P_ctrl`: the probability assigned to the *originally predicted* answer under each condition.
3. Record the **answer-flip indicator** under each condition: did top-1 predicted option change vs. the clean run?

### Step 5 — Behavioral Faithfulness Scores & Statistics
**Requirement:** Per-question score **F(x) = P_ctrl − P_ev** (large positive = faithful reliance on cited evidence; ≈0/negative = cited span no more load-bearing than arbitrary text).

**Coding tasks — in detail:**
1. `src/faithfulness.py`:
   - Compute per-question: `F = P_ctrl - P_ev`, `flip_ev` (bool), `flip_ctrl` (bool).
   - **Answer-flip rate**: % of questions whose top-1 option changes, reported *separately* for evidence-removal vs control-removal. Faithful model ⇒ visibly higher flip rate under evidence-removal.
   - **Dataset-level**: mean F(x) over N=400, plus a **Wilcoxon signed-rank test** (`scipy.stats.wilcoxon`) comparing evidence-removal drops (`P_clean − P_ev`) vs control-removal drops (`P_clean − P_ctrl`) — chosen over a paired t-test because probability-drop scores are bounded and likely non-normal.
2. Save `outputs/phase1_results.json` + a per-question CSV. These feed the Week 2 subset selection (oversample low-F cases) and the Phase 1↔2 agreement analysis.

---

## Week 2 — Phase 2 (part 1): Mechanistic Localization with Logit Lens + Start Phase 3

### Step 6 — Mechanistic Tooling Setup
**Coding tasks — in detail:**
1. Pick **one** tool and install/verify it: `TransformerLens` (hook-based, convenient for patching) **or** `nnsight` (more general). Write `src/mech_setup.py` that loads the *primary* model into the chosen library with hooks on: residual stream at every layer (all token positions), and the final unembedding projection.
2. Verify with a smoke test: run one question, cache all per-layer residual-stream activations, confirm shapes `(layers, seq_len, d_model)`.

### Step 7 — Logit Lens on the 100-Question Subset
**Requirement:** At the token position immediately preceding the final answer, project each layer's residual stream through the model's **own final unembedding matrix** (as if that layer were the last), giving a vocab distribution per layer. Read off the probability/rank of each **option letter token** (or leading token of each option text) across layers.

**What it looks for:** does the eventually-chosen option's internal probability overtake all distractors *early* (genuine internal computation) or only in the last layer or two (late, shallow resolution the CoT may be dressing up post-hoc)?

**Ground truth:** none external — it's an *observational* probe. Reference = the model's own eventual output; run on cases where the emitted answer matches MedQA gold (sanity filter).

**Coding tasks — in detail:**
1. `src/logit_lens.py`:
   - For each of the 100 questions: run the model with hooks, grab residual stream at the answer position for every layer ℓ.
   - Apply unembedding: `logits_ℓ = residual_ℓ @ W_U`, softmax → probability of each option letter token.
   - **Emergence layer** per question = first layer where the chosen option's probability exceeds *all* distractors'.
   - Mechanistic score = **earliness = 1 − (emergence_layer / total_layers)** (higher = earlier = more faithful support).
2. Save `outputs/logit_lens_100.jsonl` with per-layer option probabilities and the emergence layer.

### Step 8 — Phase 3 (start): Error Taxonomy Annotation (parallel, Weeks 2–3)
**Requirement:** Manually review **N = 50** incorrect or low-faithfulness cases (prioritize the 2×2 off-diagonal disagreement cases from Step 10 when available, else lowest-F / incorrect cases). Both team members independently label **all 50**.

**Coding tasks — in detail:**
1. `src/taxonomy.py`:
   - Define the category set (from proposal): `knowledge_gap`, `reasoning_slip`, `hallucinated_premise`, `question_misreading`, `confident_unjustified_leap` (+ `other`).
   - Build a simple annotation UI: a CSV/JSONL export of the 50 cases (id, question, trace, predicted vs gold, F(x)) loaded into a spreadsheet or a tiny Streamlit/Gradio app; each annotator writes `labels_annotator1.csv`, `labels_annotator2.csv` with columns `id, primary_label, secondary_label(optional)`.
   - One **primary label per case** (keeps κ well-defined); secondary label recorded separately, reported descriptively only.
2. Compute **inter-annotator agreement**: raw % agreement + **Cohen's κ** (`sklearn.metrics.cohen_kappa_score`), interpreted on the Landis & Koch scale (κ > 0.6 = "substantial"). Resolve disagreements by discussion; record final adjudicated label.

---

## Week 3 — Phase 2 (part 2): Activation Patching + Finish Taxonomy + Agreement Analysis

### Step 9 — Activation Patching (N = 100 subset)
**Requirement:** Causal counterpart to logit lens. For each question: run a **clean** prompt and a **corrupted** prompt (the evidence-removal vignette from Phase 1 — decisive clinical finding removed, options unchanged). Cache all clean activations; re-run corrupted but **patch in the clean activation at one targeted layer/position**; measure whether the original (clean) answer is restored.

**Ground truth:** entirely intra-example — normalized recovery score on the correct option's logit:
```
score = (logit_patched − logit_corrupted) / (logit_clean − logit_corrupted)
```
score ≈ 1 → that location is sufficient to causally restore the clean answer (trace corroborated); ≈ 0 → causally inert. Gold label enters **only as a filter**: patch only questions where the clean run matched gold.

**Coding tasks — in detail:**
1. `src/patching.py`:
   - `run_with_cache(model, prompt)` → cached activations at every (layer, token position).
   - Patching loop: for each candidate (layer, position) — start with the position(s) the stated reasoning points to, then sweep other positions for comparison — splice the clean activation into the corrupted run (TransformerLens `hook` with `cache`-replace, or nnsight intervention), read the answer-option logit at the final position.
   - Compute recovery score per patched location; identify the **minimal set of locations whose patching restores the answer** and compare with the location the trace claims.
   - Interpretation: if patching the trace-cited location does little but another location largely restores the answer → direct causal evidence the stated step isn't what the model used.
2. Save `outputs/patching_100.jsonl` (recovery score per layer/position per question).

### Step 10 — Agreement Between Behavioral and Mechanistic Signals (N = 100)
**Coding tasks — in detail:**
1. In `src/faithfulness.py` (or a new `src/agreement.py`): for each of the 100 questions join `F(x)` with the mechanistic score (logit-lens earliness and/or patching recovery, scaled so higher = more faithful).
   - **Spearman ρ** (`scipy.stats.spearmanr`) between F(x) and mechanistic score — preferred over Pearson given small N and likely non-linear relationship.
   - **Binarized agreement**: threshold both scores (median split, or F(x) > 0) into faithful/unfaithful; report raw % agreement + **Cohen's κ**; build the 2×2 contingency table.
   - Route the **off-diagonal cells** (behavioral vs mechanistic disagreement) into the Phase 3 taxonomy review — these are the most informative cases.

### Step 11 — Finish Taxonomy Annotation
Complete remaining annotations from Step 8, compute final κ, adjudicate disagreements, finalize `outputs/taxonomy_50.csv`.

---

## Week 4 — Phase 4: Synthesis & Deliverables (Steps 12–14)

### Step 12 — Consolidate Results
**Coding tasks — in detail:**
1. `src/report.py`: generate all tables/figures — flip rates (evidence vs control), mean F(x) + Wilcoxon p-value, emergence-layer distributions, patching recovery heatmaps (layer × position), Spearman ρ / κ values, 2×2 agreement table, taxonomy label distribution with κ.
2. Write the **short technical report** (markdown → PDF): faithfulness scores, localization findings, taxonomy, and discussion of implications for showing small-LM traces to clinicians/patients.

### Step 13 — Released Annotated Example Set
Package a small release: `outputs/annotated_examples.jsonl` with per-example perturbation results (P_clean, P_ev, P_ctrl, F(x), flips) + taxonomy labels for the 50 reviewed cases, plus a README describing the schema and license.

### Step 14 — Reusable Evaluation Script (key deliverable)
**Requirement:** a lightweight, standalone script for trace-perturbation faithfulness testing on MedQA-style datasets.
**Coding tasks — in detail:**
1. Refactor Steps 3–5 into a single CLI, e.g.:
   ```
   python eval_faithfulness.py \
       --model <hf-model-id> \
       --dataset medqa --n 400 --seed 42 \
       --out outputs/run1/
   ```
2. It must: load any MedQA-style MCQ dataset (question + options + gold), generate CoT traces, auto-construct evidence/control removals, compute F(x), flip rates, and the Wilcoxon test, and write results JSON/CSV. Include a `--smoke-test` mode (N=20) and a README with usage examples. No mechanistic dependencies — keep it lightweight.

---

## Deliverables Checklist (from the proposal)
- [ ] Short technical report (experiments, mechanistic findings, taxonomy)
- [ ] Small released set of annotated examples (perturbation results + taxonomy labels)
- [ ] Lightweight reusable evaluation script for trace-perturbation faithfulness on MedQA-style data

## Timeline at a Glance
| Week | Work |
|---|---|
| 1 | Setup; CoT traces on 400 MedQA; perturbation tests; behavioral F(x) scores |
| 2 | Mechanistic tooling; logit lens on 100-question subset; start taxonomy annotation |
| 3 | Activation patching; finish taxonomy; behavioral↔mechanistic agreement analysis |
| 4 | Consolidate results, figures/tables, write-up, finalize taxonomy & discussion |

## Key Risks & Mitigations
- **Phase 2 tooling difficulty** → Phase 3 taxonomy is deliberately tooling-free and can serve as fallback if mechanistic work slips.
- **Community fine-tunes lack technical reports** → vet model cards first; use one primary model, others only as robustness checks.
- **Span identification for evidence-removal is partly manual** → automate where possible, manually verify ambiguous cases; keep the control span length-matched and seeded.
- **Small samples (N=100/50)** → use rank-based (Spearman) and chance-corrected (κ) statistics rather than assuming normality.
