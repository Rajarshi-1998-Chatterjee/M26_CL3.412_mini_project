# Faithfulness of CoT Traces in Small Medical QA Models - Results

## Phase 1: Behavioral faithfulness (N=400)
- Clean accuracy: 0.3525
- Answer-flip rate, evidence-removal: 0.2625
- Answer-flip rate, control-removal: 0.4475
- Mean F(x) = P_ctrl - P_ev: -0.13606350164109401
- Wilcoxon signed-rank (ev vs ctrl drops): {'statistic': 30296.0, 'pvalue': 2.2621129438042925e-05}

## Phase 2: Mechanistic signals vs behavior (N=400)
- earliness: Spearman rho=-0.03426185097664837, p=0.8625873991524832, kappa=0.0, agreement=0.5
- patch_best: Spearman rho=0.41208791208791207, p=0.16175105277252633, kappa=0.6904761904761905, agreement=0.8461538461538461

## Phase 3: Error taxonomy
- {}

## Figures
See outputs/figures/ (fig1..fig5).
