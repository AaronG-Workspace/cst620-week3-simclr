# VAL selection results (selection only, not the reported result)

These are VALIDATION numbers. Val (173 images: 69 ok, 104 defect) chose k, C, the anomaly k,
the threshold, and the supervised best epoch, so every number here is optimistic.
The reported result is the single test run in eval.py.

Majority class (always 'defect') on val: 60.1%. One val image = 0.58 points.

## VAL accuracy by labels per class (mean +/- std over 3 draws; 'all' = 1 draw)

| method | 10 | 25 | 50 | all |
|---|---|---|---|---|
| SSL probe | 0.711 +/- 0.007 | 0.764 +/- 0.020 | 0.738 +/- 0.022 | 0.799 |
| SSL k-NN | 0.732 +/- 0.018 | 0.749 +/- 0.007 | 0.718 +/- 0.017 | 0.747 |
| random probe | 0.688 +/- 0.031 | 0.724 +/- 0.033 | 0.724 +/- 0.031 | 0.736 |
| random k-NN | 0.732 +/- 0.027 | 0.730 +/- 0.009 | 0.732 +/- 0.007 | 0.839 |
| supervised | 0.713 +/- 0.041 | 0.688 +/- 0.076 | 0.749 +/- 0.028 | 0.885 |

## Flags (VAL)

- SSL, budget 10: k-NN beats probe by 2.1 points
- random, budget 10: k-NN beats probe by 4.4 points
- random, budget 25: k-NN beats probe by 0.6 points
- random, budget 50: k-NN beats probe by 0.8 points
- random, budget all: k-NN beats probe by 10.3 points

Budgets where the probe beats k-NN by > 10 points on VAL: 0. Budgets where k-NN beats the probe on VAL: 5.

![VAL budget curve](val_budget_curve.png)
