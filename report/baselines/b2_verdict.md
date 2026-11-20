# B2 DeBERTa verdict

**Verdict: fail**

- Selected configuration: **row2**, chosen on train_pool val only (highest mean val TPR at 1% val FPR across seeds, ties by lowest mean val log-loss).
- B1 bar: TPR at 5% FPR on MCPTox = **0.2247** (`report/baselines/b1_tfidf_lr.csv`).
- A seed passes when, on MCPTox, its TPR-at-5%-FPR 95% lower bound is above the B1 bar and its length-binned AUROC 95% lower bound is above 0.5.
- Pass needs every seed to pass; fail means none did; mixed is anything between. Nothing is rerun after this verdict.

## Selection on val

| row | seeds | mean val TPR @1% FPR | mean val log-loss | mean val AUROC |
|---|---|---|---|---|
| row1 | 3 | 0.9984 | 0.037990 | 0.9998 |
| row2 (selected) | 3 | 0.9995 | 0.029038 | 0.9999 |
| row3 | 3 | 0.6911 | 1.066016 | 0.9622 |

## row2 on MCPTox, every seed

| seed | TPR @5% FPR | lower | upper | length-binned AUROC | lower | upper | passes |
|---|---|---|---|---|---|---|---|
| 7267 | 0.2082 | 0.1526 | 0.2495 | 0.7085 | 0.6555 | 0.7703 | no |
| 7268 | 0.0392 | 0.0227 | 0.0557 | 0.3902 | 0.3416 | 0.4413 | no |
| 7269 | 0.0412 | 0.0247 | 0.0742 | 0.4900 | 0.4399 | 0.5468 | no |

## Appendix: val-only sweep runs

Not reported runs: val numbers only, never scored on the holdouts.

| run | row | seed | val TPR @1% FPR | val log-loss | val AUROC |
|---|---|---|---|---|---|
| sweep-row2-w0.5-s7267 | row2 | 7267 | 0.9984 | 0.048059 | 0.9999 |
| sweep-row2-w1.0-s7267 | row2 | 7267 | 0.9984 | 0.057430 | 0.9999 |
