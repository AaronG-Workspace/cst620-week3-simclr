# TEST results (frozen choices)

Frozen at 2026-09-29T00:56:25+00:00; every hash verified before scoring.
260 images: 104 ok, 156 defect. Majority class: 60.0%. One image = 0.38 points.
Strata: near 58, clean 202 (near = a test image whose nearest train image by random-init feature cosine is within 8-orientation 256-bit pHash distance <= 40; clean = every other test image.)

## Accuracy (mean +/- std over 3 draws; 'all' = 1 draw)

| method | 10 | 25 | 50 | all |
|---|---|---|---|---|
| SSL k-NN | 0.656 +/- 0.025 | 0.681 +/- 0.023 | 0.638 +/- 0.044 | 0.758 |
| SSL probe | 0.621 +/- 0.002 | 0.676 +/- 0.056 | 0.738 +/- 0.042 | 0.912 |
| random k-NN | 0.626 +/- 0.021 | 0.655 +/- 0.049 | 0.662 +/- 0.050 | 0.877 |
| random probe | 0.619 +/- 0.038 | 0.658 +/- 0.028 | 0.676 +/- 0.028 | 0.769 |
| supervised | 0.636 +/- 0.024 | 0.564 +/- 0.036 | 0.668 +/- 0.027 | 0.835 |

## Probe vs k-NN flags

- SSL, budget 10: k-NN beats probe by 3.6 points
- SSL, budget 25: k-NN beats probe by 0.5 points
- SSL, budget 50: probe beats k-NN by 10.0 points (> 10)
- SSL, budget all: probe beats k-NN by 15.4 points (> 10)
- random, budget 10: k-NN beats probe by 0.6 points
- random, budget all: k-NN beats probe by 10.8 points

## Per-class recall (mean over draws): ok / defect

| method | 10 | 25 | 50 | all |
|---|---|---|---|---|
| SSL k-NN | 0.465 / 0.784 | 0.545 / 0.771 | 0.458 / 0.759 | 0.625 / 0.846 |
| SSL probe | 0.369 / 0.788 | 0.503 / 0.791 | 0.622 / 0.816 | 0.904 / 0.917 |
| random k-NN | 0.362 / 0.801 | 0.449 / 0.793 | 0.474 / 0.786 | 0.837 / 0.904 |
| random probe | 0.497 / 0.701 | 0.465 / 0.786 | 0.490 / 0.799 | 0.644 / 0.853 |
| supervised | 0.372 / 0.812 | 0.481 / 0.620 | 0.439 / 0.821 | 0.712 / 0.917 |

## Anomaly score (frozen option and threshold; 95% CI = group bootstrap, 1000 resamples, seed 42)

| encoder | budget | draw | option | AUROC [95% CI] | missed defects | false rejects | defect recall | ok recall | AUROC near / clean |
|---|---|---|---|---|---|---|---|---|---|
| SSL | 10 | 0 | 10 | 0.719 [0.592, 0.870] | 2 | 89 | 0.987 | 0.144 | 0.710 / 0.701 |
| SSL | 25 | 0 | all_ok | 0.733 [0.602, 0.863] | 11 | 79 | 0.929 | 0.240 | 0.666 / 0.727 |
| SSL | 50 | 0 | all_ok | 0.727 [0.598, 0.874] | 2 | 87 | 0.987 | 0.163 | 0.707 / 0.712 |
| SSL | 10 | 1 | 10 | 0.683 [0.558, 0.829] | 4 | 93 | 0.974 | 0.106 | 0.708 / 0.661 |
| SSL | 25 | 1 | all_ok | 0.673 [0.548, 0.832] | 4 | 92 | 0.974 | 0.115 | 0.680 / 0.657 |
| SSL | 50 | 1 | all_ok | 0.702 [0.573, 0.854] | 0 | 91 | 1.000 | 0.125 | 0.710 / 0.681 |
| SSL | 10 | 2 | 10 | 0.713 [0.580, 0.867] | 2 | 79 | 0.987 | 0.240 | 0.715 / 0.693 |
| SSL | 25 | 2 | all_ok | 0.739 [0.608, 0.876] | 5 | 80 | 0.968 | 0.231 | 0.703 / 0.727 |
| SSL | 50 | 2 | all_ok | 0.699 [0.571, 0.853] | 0 | 93 | 1.000 | 0.106 | 0.709 / 0.678 |
| SSL | all | 0 | all_ok | 0.706 [0.577, 0.862] | 2 | 89 | 0.987 | 0.144 | 0.709 / 0.686 |
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
| SSL k-NN | 0.603 / 0.672 | 0.649 / 0.690 | 0.615 / 0.645 | 0.776 / 0.752 |
| SSL probe | 0.580 / 0.632 | 0.626 / 0.690 | 0.707 / 0.748 | 0.914 / 0.911 |
| random k-NN | 0.552 / 0.647 | 0.638 / 0.660 | 0.638 / 0.668 | 0.948 / 0.856 |
| random probe | 0.644 / 0.612 | 0.632 / 0.665 | 0.667 / 0.678 | 0.776 / 0.767 |
| supervised | 0.621 / 0.640 | 0.500 / 0.583 | 0.615 / 0.683 | 0.845 / 0.832 |

## Embedding quality (one image per group, seed 42; L2-normalized)

| encoder | silhouette | Davies-Bouldin | n |
|---|---|---|---|
| SSL | 0.2077 | 2.2066 | 160 |
| random-init | 0.0382 | 4.9085 | 160 |
| supervised (all) | 0.2430 | 1.4819 | 160 |

Higher silhouette and lower Davies-Bouldin = cleaner ok/defect separation. The t-SNE plot (tsne.png) is a picture, not proof; the k-NN, probe and anomaly numbers are the proof.
