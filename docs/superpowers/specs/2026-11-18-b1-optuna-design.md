# B1-Optuna: searching B1's features and regularization by family-grouped CV

Date: 2026-11-18
Status: run 2026-11-18; result: better than B1 under the decision rule (see the plan, Task 5)
Serves: Phase 4 task 2.2 (hyperparameter search), and the "word vs character n-grams" row of
the task 4 ablation table.

## Goal

B1 TF-IDF searches only `C ∈ {0.1, 1, 10}` on val AUROC and picks the edge value, 10. Its
features (word 1–2-grams plus `char_wb` 3–5-grams, `min_df=2`, `sublinear_tf`) are fixed.
This spec adds an Optuna study over B1's features and regularization, scored on attack
families held out of fitting, and reports the result as a **separate B1 variant**,
`tfidf-optuna`, next to B1 and B1-famcv. B1 itself is unchanged.

**An honest null result is fine.** If the study does not beat B1 under the decision rule
below, the write-up says so (Phase 4 task 7.6).

## Why a separate variant

The B1-famcv spec lists "extending B1's `C` grid" as a non-goal, because it would be a second
variable next to the selection method. Running the study as its own variant keeps both specs
intact: B1 vs B1-famcv isolates the selection method, and B1-famcv vs B1-Optuna isolates the
search space (both use family folds of train).

## Decisions

| # | Decision | Choice |
|---|---|---|
| 1 | Objective | Mean held-out AUROC over the 5 family folds of **train_pool train** (`family_folds` from the B1-famcv plan, `k=5`, `seed=7267`). Val is not used by the study |
| 2 | Folds | Identical to `tfidf-famcv`: BIPIA families bucketed greedily, PoisonedRAG groups dealt round-robin, twins never split |
| 3 | Sampler | `TPESampler(seed=7267)`, trials run one at a time, so the study is reproducible |
| 4 | Budget | 100 trials, fixed before the run |
| 5 | Pruning | `MedianPruner(n_startup_trials=10, n_warmup_steps=1)` on the running mean AUROC after each fold. Fold order is the same for every trial, so running means are comparable |
| 6 | Search space | `C` log-uniform 1e-2 to 1e3; features `word`, `char` or `both`; word n-grams (1, n) with n ∈ {1, 2, 3}; char analyzer `char` or `char_wb`; char range (2,4), (3,5) or (3,6); `min_df` ∈ {1, 2, 5}; `max_features` ∈ {unlimited, 50k, 200k} per vectorizer; `sublinear_tf` on or off; `class_weight` none or `balanced`. Word and char settings are only sampled when that feature set is used |
| 7 | Final model | The best trial's settings, refit on all of train. The threshold is set at 1% FPR on val's benign rows, exactly as in B1 |
| 8 | Evaluation | Once, through the unchanged harness: test, MCPTox, MSB, same columns as every other baseline CSV |
| 9 | Detector name | `tfidf-optuna` (report `detector` column and MLflow run name) |
| 10 | Holdout guard | The study's rows come from `dataset.load("fit", roles=["train_pool"], splits=["train"])`, and the study itself refuses any row that is not train_pool train (`HoldoutError`), so val, test, MCPTox and MSB cannot reach a trial |
| 11 | B1's code | `src/models/tfidf.py` is not edited (it is untouchable under the famcv plan). The configurable detector subclasses it, and its defaults reproduce B1 exactly (tested) |

## Outputs

- `report/baselines/b1_tfidf_lr_optuna.csv`: same columns as `b1_tfidf_lr.csv`.
- `report/phase4/b1_optuna/`: `best_params.json`, `trials.csv`, `folds.csv` (fold sizes by
  source and label), `history.png`, `importance.png`.
- MLflow (when a tracking URI is set): one `tfidf-optuna` run in `phase3-detectors` with the
  same params and metrics as other detectors, plus one nested run per trial holding only its
  parameters, state and score.

## Decision rule (fixed before the run)

Measured on the **train_pool test split only**. `tfidf-optuna` is reported as better than B1
if its test `all` AUROC is above B1's upper 95% bootstrap bound, or if its test `all` AUROC is
no more than 0.01 below B1's and its `source=bipia` AUROC is above B1's. Otherwise it is a
null result. MCPTox and MSB are reported for every variant and never used to choose.

## Anti-goals

- Any val, test, MCPTox or MSB row in a trial.
- Changing the search space, budget, sampler seed or folds after seeing any evaluation number.
- Reporting `tfidf-optuna` without B1 beside it.
