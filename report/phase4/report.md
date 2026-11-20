---
title: "Detecting Prompt Injection Across RAG and MCP Surfaces"
subtitle: "Phase 4 Report: Model Development and Experimentation"
author:
  - Carter Corbin
  - Max Edmonds
  - Mohammed Adil Ahmed
  - Oluwamayowa Adewummi
  - Adedapo Odeyemi
date: |
  ```{=latex}
  \vspace{-0.6em}College of Computing and Software Engineering\\
  Kennesaw State University, Kennesaw, GA\\[1.2em]
  CS 7267 Machine Learning \textperiodcentered{} December 2026
  ```
abstract: |
  TODO (write last, ~150 words). Draft claim: three detector families plus a length
  reference, trained on RAG passages and tool outputs, all fit their training surfaces but
  none transfers to MCP tool descriptions. A tuned TF-IDF model (Optuna, family-grouped CV)
  is the strongest trained detector on both holdouts, yet raw length still matches it on
  MCPTox AUROC. A fine-tuned DeBERTa-v3 reaches 0.99 test AUROC but scores below chance on
  MCPTox and fails its pre-registered criterion on every seed. The evidence points to a genre
  gap, not a threshold or capacity problem.
---

<!--
OUTLINE STATUS (delete this block before submission)

Numbers freeze 2026-11-30; write-up due 2026-12-07. Build like Phase 2 (pandoc + tectonic;
copy header.tex, ieee.csl, repo-links.lua and extra.bib from report/phase2).

Each section below lists what it must say, the evidence for it, and its status:
  [ready]    results exist on main or in an open PR; write now
  [pending]  depends on unfinished Phase 4 work; the task number is from
             docs/phase4-requirements-tasks.md
  [decide]   needs a team decision first

Open PRs this report depends on: #40 (shortcut gate, red team), #44 (B2 seed summary),
#45 (B1-Optuna). Owners for each section: TBD at the next team meeting.
-->

# Introduction

[ready] ~0.75 page.

- Problem recap from Phase 2: indirect prompt injection in text an LLM reads but the user did
  not write (RAG passages, tool outputs, MCP tool descriptions).
- The gaps from Phase 2 this phase tests: no detector evaluated on an unseen attack surface;
  none on MCP tool descriptions; the length shortcut every detector must be measured against.
- Phase 4 question: does a stronger, tuned detector trained on two surfaces transfer to a
  third it never saw?
- Contributions (one line each): three detector families plus length under one harness; a
  hyperparameter search scored on held-out attack families; a pre-registered success criterion
  and its verdict; an ablation of the design choices; an honest account of what failed.

# Experimental Setup

## Data and splits

[ready] Recap only; point to the Phase 2 data card.

| Split | Role | Benign | Adversarial | Used for |
|---|---|---|---|---|
| train_pool train | fit | 4,133 | 3,084 | fitting, Optuna folds |
| train_pool val | fit | 801 | 615 | model selection, deployed threshold (1% FPR) |
| train_pool test | eval | 827 | 607 | in-distribution results, decision rules |
| MCPTox | LOSO holdout | 362 | 485 | unseen surface (tool descriptions), scored once per run |
| MSB | transfer holdout | 21 | 70 | cross-benchmark, indicative only |

- Twins: every BIPIA context appears clean and attacked in one `group`; groups never span
  splits or folds.
- `attack_kind` mix: train pool has poisoning and instruction injection; both holdouts are
  instruction-only.

## Evaluation protocol

[ready]

- One harness for every detector (`src/eval/harness.py`); every number in this report comes
  from it.
- Metrics: AUROC with a 95% bootstrap CI; deployed TPR and FPR at the threshold set at 1% FPR
  on val benign rows and applied unchanged; TPR at a matched 5% FPR on each eval set's own
  benign rows (measurement only); length-binned AUROC (is the signal more than length?).
- Rules fixed before any run: holdouts never used to select a model, hyperparameter, seed or
  threshold; every reported run is shown; decision rules use train_pool test only.
- Wilson intervals for FPR in the tables. [pending 3.4]

## Reproducibility and tracking

[pending 5.1–5.5, 8.x] What exists vs what is promised.

- Exists: DVC for data and score files on R2; MLflow (`phase3-detectors`, shared Neon
  backend) for every reported run; seeds fixed; `make evaluate`, `make b1-optuna`,
  `make b2-report`.
- Pending: one config file for all experiments (5.1–5.2), `make reproduce` (5.3), DVC stages
  for evaluation (5.4). B2 training is the one manual step (Colab notebook).
- Mention: the MLflow outage on 2026-11-19 (Neon compute removed) and the `--log-only` path
  that recovered it without re-running.

# Model Families

[ready, except 3.5] ~1 page. One paragraph each: what it is, why it is in the study, what was
tuned.

## Length-only reference
- Score = text length. The bar from Phase 2: test 0.491, MCPTox 0.774, MSB 0.866 AUROC.

## B0: perplexity (frozen DistilGPT2)
- Unsupervised, not trained on labels, not tuned. Near chance in distribution (0.515), below
  chance on the holdouts (MCPTox 0.430, MSB 0.134).

## B1: TF-IDF + logistic regression
- Word 1–2 + `char_wb` 3–5-grams; `C` chosen on val from {0.1, 1, 10} (picked 10, the edge).
- Variants: B1-Optuna (Section 4.1); B1-famcv (Section 5.2) [pending 3.1].

## B2: fine-tuned DeBERTa-v3-base
- Chunked max-pool scoring (overlapping 256-token windows, score = max window logit).
- Pairwise margin loss on BIPIA clean/attacked twins.
- Selection on val TPR at 1% FPR, ties by val log-loss (amendment 2026-11-08, val saturated).
- Trained on Colab T4, fp16, gradient clipping at 1.0 (seed 7267 of row 1 predates clipping).

## Third trained family
[decide: risk R1] B0 is not trained. Either add a trained family (e.g. LightGBM on TF-IDF and
length features) through the same harness, or argue in one paragraph why B0 counts. Cut this
subsection if the team argues B0.

# Hyperparameter Search

## B1-Optuna
[ready, PR #45] ~0.75 page plus one figure.

- Search space: `C` log-uniform 1e-2..1e3; word, char or both; n-gram ranges; char analyzer;
  `min_df`; `max_features`; `sublinear_tf`; `class_weight`.
- Objective: mean held-out AUROC over 5 family folds of train only (unseen attack families);
  TPE, seed 7267, 100 trials, median pruner. Val and holdouts never seen by a trial (tested).
- Result: best trial #93, mean CV AUROC 0.9215; word 1–3 + `char` 2–4-grams, `C` 6.38,
  `min_df` 1, 200k features, `class_weight="balanced"`.
- Decision rule (fixed beforehand): better than B1. Test AUROC 0.922 [0.907, 0.935] vs
  0.902 [0.886, 0.917].
- Figure: `report/phase4/b1_optuna/importance.png` (or history.png). Note: the pruner stopped
  no trial.

## B2 search
[ready] Short.

- Two-point `pairwise_weight` sweep {0.5, 1.0} on val, one seed; 0.5 chosen. Val saturates
  (AUROC ≈ 1.0 from epoch 1), so val cannot rank configurations well (amendment).
- B2 Optuna study (task 2.3): [decide] report as not run within the Colab budget, or run 6–10
  val-only trials before Nov 30.

## B0
[decide 2.4] State as an untuned reference and why (frozen LM, no labels), or search
`max_context`/`stride` on val.

# Cross-Validation and Uncertainty

## Family-grouped folds
[ready] Design only: BIPIA families bucketed whole into 5 folds, PoisonedRAG groups dealt
round-robin, twins never split. Fold table from `report/phase4/b1_optuna/folds.csv`.

## B1-famcv
[pending 3.1–3.2] Choosing `C` (and, in one variant, the threshold) by family-grouped CV;
per-fold table with mean and t-interval; decision rule from the famcv spec. Null result is
acceptable and expected for MCPTox FPR.

## Seed variance for B2
[ready, PR #44] Three seeds per row; report mean ± std (`report/baselines/b2_seed_summary.md`)
next to per-run bootstrap CIs. k-fold training is out of Colab budget (state why).

# Results

[ready for B0, B1, B1-Optuna, B2; pending for B1-famcv and any third family]
One table per eval set (B2 spec deliverable): length, B0, B1, B1-Optuna, B1-famcv, B2 rows.
Columns: AUROC [CI], length-binned AUROC, deployed TPR, deployed FPR [Wilson], TPR at matched
5% FPR.

## In-distribution (train_pool test)
- Every trained model fits: B1 0.902, B1-Optuna 0.922, B2 rows 0.995 / 0.987 / 0.936 (means).

## MCPTox (unseen surface)
- Best trained detector: B1-Optuna, 0.767 [0.736, 0.800] AUROC, which only ties raw length
  (0.774). B1 0.675; B2 rows 0.36 / 0.35 / 0.20 (below chance); B0 0.430.
- B1-Optuna's length-binned AUROC (0.725) shows its signal is more than length, which length
  itself cannot claim; say so carefully.
- Deployed FPR drifts above 1% for every detector (genre gap, Section 9).

## MSB (transfer, indicative)
- 21 benign rows: one false positive moves FPR by 4.8 points; label as indicative.

## B2 verdict under the pre-registered criterion
- **Fail.** Row 2 selected on val. On MCPTox no seed clears B1's TPR at 5% FPR (0.2247):
  seeds 0.208 / 0.039 / 0.041, and only seed 7267 has a length-binned AUROC lower bound above
  0.5. Source: `report/baselines/b2_verdict.md`.
- B2's bar stays the original B1, as fixed before B2 ran. [decide: team confirms]

# Ablation Study

[ready for B2 rows and B1 vs B1-Optuna; pending for famcv rows] One table: each row changes
one thing, same eval sets and interval columns, length-binned AUROC included (task 4.1–4.2).

| Design choice | Comparison | Status |
|---|---|---|
| Pairwise loss on twins | B2 row 1 vs row 2 | ready |
| PoisonedRAG in training | B2 row 2 vs row 3 | ready |
| Search space and CV objective | B1 vs B1-Optuna | ready |
| Word vs character n-grams | Optuna trials grouped by `features` | ready (from `trials.csv`) |
| Selection on val vs family CV | B1 vs B1-famcv | pending 3.1 |
| Threshold from val vs out-of-fold | B1-famcv variants | pending 3.1 |
| Synthetic tool descriptions | B2 row 4 | cut (stretch, out of budget) |

# Robustness and Shortcut Analysis

[pending merge of PR #40]

- Shortcut gate: length alone beats B1 on both holdouts (gate FAIL). Run the gate on
  B1-Optuna and B2 too.
- Red-team mutations of the test split: B1's TPR falls from 0.647 to 0.280 with leetspeak,
  0.328 with zero-width characters, 0.516 with homoglyphs; rewording and reordering change
  little. [decide: score the mutations with B1-Optuna and B2 before Nov 30?]

# Negative Results and Limitations

[ready] Task 7 list, each with numbers.

- B2 fails its criterion on every seed; fits training surfaces (test AUROC ≈ 0.99) but scores
  below chance on MCPTox. Row 2 is unstable across seeds (MCPTox AUROC 0.25–0.49; val
  threshold 0.006–0.989).
- Val saturation: B2 val AUROC ≈ 1.0 from epoch 1, so val cannot rank B2 configurations.
- B0 below chance on the holdouts, near chance in distribution.
- MCPTox FPR is a genre gap, not a threshold problem: train has no benign tool descriptions.
- Small holdouts: MSB has 21 benign rows; MSB `tool_output` has no benign rows and no FPR.
- Search limits: B2 search is two points; the Optuna pruner never pruned; B1-famcv null
  result if it picks the same `C` [pending 3.1].
- Process: seed 7267 of B2 row 1 predates gradient clipping (not retrained, by rule).

# Discussion and Next Steps

[ready to draft once Results is final] ~0.5 page.

- What the evidence says: capacity is not the bottleneck (DeBERTa fits; TF-IDF transfers
  better); exposure to the target genre is.
- What would change it: benign and adversarial tool descriptions in training (B2 stream 5,
  cut); evaluating end-to-end attack success in AgentDojo.
- Future-work notes: skill registries as an injection surface (`report/future-work/`).

**Use of AI assistance.** [ready] Mirror Phase 2's statement: what was drafted or written
with an AI assistant (code through reviewed PRs, analysis scripts, this outline), and that the
team reviewed every analysis choice and the final text.

\appendix

# Full Per-Slice Results {#appendix-slices}

[ready] Per-slice tables (source, attack surface, attack kind) for each detector, from
`report/baselines/*.csv`.

# B2 Val-Only Sweep Runs {#appendix-sweeps}

[ready] The `sweep-row2-w0.5/w1.0-s7267` table from `b2_verdict.md`.

# Reproducing the Results {#appendix-reproduce}

[pending 5.3] The command sequence (`make install`, `make data`, `make reproduce`), the Colab
step for B2 training, and where MLflow runs live.
