# CONTAMINATED TEST results (frozen choices)

**CONTAMINATED. The SSL encoder was pretrained on the train pool PLUS the TEST images.** Leakage control only, never a result. Same frozen k, C, anomaly option and threshold as the clean run; random-init and supervised rows are unchanged from the clean run. The leakage estimate is in leakage_gap.json (CONTAMINATED minus clean).

Frozen at 2026-09-29T00:56:25+00:00; every hash verified before scoring.
260 images: 104 ok, 156 defect. Majority class: 60.0%. One image = 0.38 points.
Strata: near 58, clean 202 (near = a test image whose nearest train image by random-init feature cosine is within 8-orientation 256-bit pHash distance <= 40; clean = every other test image.)

## Accuracy (mean +/- std over 3 draws; 'all' = 1 draw)

| method | 10 | 25 | 50 | all |
|---|---|---|---|---|
| SSL k-NN | 0.649 +/- 0.030 | 0.678 +/- 0.021 | 0.638 +/- 0.043 | 0.758 |
| SSL probe | 0.638 +/- 0.018 | 0.674 +/- 0.039 | 0.692 +/- 0.047 | 0.915 |
| random k-NN | 0.626 +/- 0.021 | 0.655 +/- 0.049 | 0.662 +/- 0.050 | 0.877 |
| random probe | 0.619 +/- 0.038 | 0.658 +/- 0.028 | 0.676 +/- 0.028 | 0.769 |
| supervised | 0.636 +/- 0.024 | 0.564 +/- 0.036 | 0.668 +/- 0.027 | 0.835 |

## Probe vs k-NN flags

- SSL, budget 10: k-NN beats probe by 1.0 points
- SSL, budget 25: k-NN beats probe by 0.4 points
- SSL, budget all: probe beats k-NN by 15.8 points (> 10)
- random, budget 10: k-NN beats probe by 0.6 points
- random, budget all: k-NN beats probe by 10.8 points

## Per-class recall (mean over draws): ok / defect

| method | 10 | 25 | 50 | all |
|---|---|---|---|---|
| SSL k-NN | 0.548 / 0.716 | 0.561 / 0.756 | 0.487 / 0.739 | 0.692 / 0.801 |
| SSL probe | 0.471 / 0.750 | 0.529 / 0.771 | 0.571 / 0.774 | 0.904 / 0.923 |
| random k-NN | 0.362 / 0.801 | 0.449 / 0.793 | 0.474 / 0.786 | 0.837 / 0.904 |
| random probe | 0.497 / 0.701 | 0.465 / 0.786 | 0.490 / 0.799 | 0.644 / 0.853 |
| supervised | 0.372 / 0.812 | 0.481 / 0.620 | 0.439 / 0.821 | 0.712 / 0.917 |

## Anomaly score (frozen option and threshold; 95% CI = group bootstrap, 1000 resamples, seed 42)

| encoder | budget | draw | option | AUROC [95% CI] | missed defects | false rejects | defect recall | ok recall | AUROC near / clean |
|---|---|---|---|---|---|---|---|---|---|
| SSL | 10 | 0 | 10 | 0.712 [0.568, 0.871] | 20 | 65 | 0.872 | 0.375 | 0.720 / 0.688 |
| SSL | 25 | 0 | all_ok | 0.717 [0.573, 0.867] | 24 | 64 | 0.846 | 0.385 | 0.704 / 0.696 |
| SSL | 50 | 0 | all_ok | 0.714 [0.570, 0.872] | 20 | 66 | 0.872 | 0.365 | 0.720 / 0.690 |
| SSL | 10 | 1 | 10 | 0.691 [0.566, 0.838] | 12 | 70 | 0.923 | 0.327 | 0.704 / 0.670 |
| SSL | 25 | 1 | all_ok | 0.699 [0.579, 0.838] | 10 | 76 | 0.936 | 0.269 | 0.702 / 0.683 |
| SSL | 50 | 1 | all_ok | 0.694 [0.548, 0.856] | 14 | 68 | 0.910 | 0.346 | 0.712 / 0.666 |
| SSL | 10 | 2 | 10 | 0.702 [0.550, 0.866] | 16 | 70 | 0.897 | 0.327 | 0.716 / 0.677 |
| SSL | 25 | 2 | all_ok | 0.723 [0.577, 0.870] | 22 | 65 | 0.859 | 0.375 | 0.707 / 0.702 |
| SSL | 50 | 2 | all_ok | 0.702 [0.562, 0.860] | 11 | 69 | 0.929 | 0.337 | 0.714 / 0.677 |
| SSL | all | 0 | all_ok | 0.700 [0.556, 0.862] | 13 | 68 | 0.917 | 0.346 | 0.715 / 0.673 |
| random | 10 | 0 | centroid | 0.669 [0.591, 0.759] | 10 | 84 | 0.936 | 0.192 | 0.611 / 0.686 |
| random | 25 | 0 | all_ok | 0.686 [0.599, 0.775] | 5 | 91 | 0.968 | 0.125 | 0.636 / 0.698 |
| random | 50 | 0 | all_ok | 0.689 [0.609, 0.763] | 7 | 86 | 0.955 | 0.173 | 0.612 / 0.707 |
| random | 10 | 1 | 10 | 0.622 [0.522, 0.702] | 7 | 92 | 0.955 | 0.115 | 0.549 / 0.643 |
| random | 25 | 1 | all_ok | 0.641 [0.559, 0.729] | 10 | 93 | 0.936 | 0.106 | 0.587 / 0.662 |
| random | 50 | 1 | all_ok | 0.657 [0.579, 0.745] | 12 | 86 | 0.923 | 0.173 | 0.590 / 0.678 |
| random | 10 | 2 | 5 | 0.656 [0.581, 0.759] | 9 | 87 | 0.942 | 0.163 | 0.606 / 0.666 |
| random | 25 | 2 | 20 | 0.621 [0.540, 0.705] | 6 | 89 | 0.962 | 0.144 | 0.547 / 0.644 |
| random | 50 | 2 | all_ok | 0.648 [0.566, 0.733] | 11 | 88 | 0.929 | 0.154 | 0.582 / 0.670 |
| random | all | 0 | 1 | 0.922 [0.855, 0.967] | 16 | 29 | 0.897 | 0.721 | 0.954 / 0.911 |

## Strata: accuracy near (58) / clean (202), mean over draws

| method | 10 | 25 | 50 | all |
|---|---|---|---|---|
| SSL k-NN | 0.632 / 0.653 | 0.672 / 0.680 | 0.615 / 0.645 | 0.828 / 0.738 |
| SSL probe | 0.615 / 0.645 | 0.644 / 0.683 | 0.667 / 0.700 | 0.897 / 0.921 |
| random k-NN | 0.552 / 0.647 | 0.638 / 0.660 | 0.638 / 0.668 | 0.948 / 0.856 |
| random probe | 0.644 / 0.612 | 0.632 / 0.665 | 0.667 / 0.678 | 0.776 / 0.767 |
| supervised | 0.621 / 0.640 | 0.500 / 0.583 | 0.615 / 0.683 | 0.845 / 0.832 |

## Embedding quality (one image per group, seed 42; L2-normalized)

| encoder | silhouette | Davies-Bouldin | n |
|---|---|---|---|
| SSL | 0.1705 | 2.1664 | 160 |
| random-init | 0.0382 | 4.9085 | 160 |
| supervised (all) | 0.2430 | 1.4819 | 160 |

Higher silhouette and lower Davies-Bouldin = cleaner ok/defect separation. The t-SNE plot (tsne.png) is a picture, not proof; the k-NN, probe and anomaly numbers are the proof.
