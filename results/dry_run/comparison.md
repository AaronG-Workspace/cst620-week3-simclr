# DRY RUN (VAL) results (frozen choices)

**DRY RUN on VAL. Not the reported result.** Val chose these settings, so these numbers are optimistic.

Frozen at 2026-09-29T00:56:25+00:00; every hash verified before scoring.
174 images: 69 ok, 105 defect. Majority class: 60.3%. One image = 0.57 points.
Strata: near 35, clean 139 (near = a test image whose nearest train image by random-init feature cosine is within 8-orientation 256-bit pHash distance <= 40; clean = every other test image.)

## Accuracy (mean +/- std over 3 draws; 'all' = 1 draw)

| method | 10 | 25 | 50 | all |
|---|---|---|---|---|
| SSL k-NN | 0.738 +/- 0.041 | 0.743 +/- 0.027 | 0.711 +/- 0.023 | 0.707 |
| SSL probe | 0.713 +/- 0.030 | 0.745 +/- 0.054 | 0.761 +/- 0.046 | 0.799 |
| random k-NN | 0.732 +/- 0.033 | 0.730 +/- 0.011 | 0.732 +/- 0.009 | 0.839 |
| random probe | 0.688 +/- 0.038 | 0.724 +/- 0.040 | 0.724 +/- 0.038 | 0.736 |
| supervised | 0.713 +/- 0.050 | 0.688 +/- 0.093 | 0.749 +/- 0.035 | 0.885 |

## Probe vs k-NN flags

- SSL, budget 10: k-NN beats probe by 2.5 points
- random, budget 10: k-NN beats probe by 4.4 points
- random, budget 25: k-NN beats probe by 0.6 points
- random, budget 50: k-NN beats probe by 0.8 points
- random, budget all: k-NN beats probe by 10.3 points

## Per-class recall (mean over draws): ok / defect

| method | 10 | 25 | 50 | all |
|---|---|---|---|---|
| SSL k-NN | 0.715 / 0.752 | 0.763 / 0.730 | 0.686 / 0.727 | 0.565 / 0.800 |
| SSL probe | 0.609 / 0.781 | 0.696 / 0.778 | 0.710 / 0.794 | 0.565 / 0.952 |
| random k-NN | 0.643 / 0.790 | 0.676 / 0.765 | 0.715 / 0.743 | 0.696 / 0.933 |
| random probe | 0.710 / 0.673 | 0.686 / 0.749 | 0.662 / 0.765 | 0.536 / 0.867 |
| supervised | 0.594 / 0.790 | 0.720 / 0.667 | 0.662 / 0.806 | 0.754 / 0.971 |

## Anomaly score (frozen option and threshold; 95% CI = group bootstrap, 1000 resamples, seed 42)

| encoder | budget | draw | option | AUROC [95% CI] | missed defects | false rejects | defect recall | ok recall | AUROC near / clean |
|---|---|---|---|---|---|---|---|---|---|
| SSL | 10 | 0 | 10 | 0.803 [0.607, 0.906] | 5 | 49 | 0.952 | 0.290 | 0.895 / 0.786 |
| SSL | 25 | 0 | all_ok | 0.806 [0.632, 0.905] | 5 | 48 | 0.952 | 0.304 | 0.878 / 0.796 |
| SSL | 50 | 0 | all_ok | 0.810 [0.624, 0.910] | 5 | 48 | 0.952 | 0.304 | 0.891 / 0.794 |
| SSL | 10 | 1 | 10 | 0.784 [0.553, 0.903] | 5 | 44 | 0.952 | 0.362 | 0.861 / 0.771 |
| SSL | 25 | 1 | all_ok | 0.790 [0.592, 0.895] | 5 | 47 | 0.952 | 0.319 | 0.857 / 0.774 |
| SSL | 50 | 1 | all_ok | 0.786 [0.575, 0.899] | 5 | 48 | 0.952 | 0.304 | 0.881 / 0.767 |
| SSL | 10 | 2 | 10 | 0.789 [0.599, 0.893] | 5 | 53 | 0.952 | 0.232 | 0.867 / 0.773 |
| SSL | 25 | 2 | all_ok | 0.813 [0.638, 0.908] | 5 | 48 | 0.952 | 0.304 | 0.891 / 0.798 |
| SSL | 50 | 2 | all_ok | 0.785 [0.574, 0.897] | 5 | 50 | 0.952 | 0.275 | 0.878 / 0.765 |
| SSL | all | 0 | all_ok | 0.787 [0.580, 0.897] | 5 | 51 | 0.952 | 0.261 | 0.878 / 0.768 |
| random | 10 | 0 | centroid | 0.607 [0.489, 0.764] | 5 | 60 | 0.952 | 0.130 | 0.721 / 0.582 |
| random | 25 | 0 | all_ok | 0.627 [0.514, 0.772] | 5 | 62 | 0.952 | 0.101 | 0.769 / 0.597 |
| random | 50 | 0 | all_ok | 0.585 [0.446, 0.774] | 5 | 60 | 0.952 | 0.130 | 0.718 / 0.555 |
| random | 10 | 1 | 10 | 0.544 [0.400, 0.746] | 5 | 65 | 0.952 | 0.058 | 0.633 / 0.526 |
| random | 25 | 1 | all_ok | 0.623 [0.518, 0.769] | 5 | 61 | 0.952 | 0.116 | 0.714 / 0.603 |
| random | 50 | 1 | all_ok | 0.619 [0.507, 0.770] | 5 | 60 | 0.952 | 0.130 | 0.718 / 0.597 |
| random | 10 | 2 | 5 | 0.637 [0.523, 0.791] | 5 | 60 | 0.952 | 0.130 | 0.711 / 0.618 |
| random | 25 | 2 | 20 | 0.572 [0.447, 0.755] | 5 | 63 | 0.952 | 0.087 | 0.650 / 0.555 |
| random | 50 | 2 | all_ok | 0.607 [0.491, 0.768] | 5 | 62 | 0.952 | 0.101 | 0.704 / 0.586 |
| random | all | 0 | 1 | 0.717 [0.579, 0.935] | 5 | 31 | 0.952 | 0.551 | 0.966 / 0.658 |

## Strata: accuracy near (35) / clean (139), mean over draws

| method | 10 | 25 | 50 | all |
|---|---|---|---|---|
| SSL k-NN | 0.810 / 0.719 | 0.800 / 0.729 | 0.781 / 0.693 | 0.914 / 0.655 |
| SSL probe | 0.762 / 0.700 | 0.800 / 0.731 | 0.810 / 0.748 | 0.886 / 0.777 |
| random k-NN | 0.800 / 0.715 | 0.800 / 0.712 | 0.819 / 0.710 | 1.000 / 0.799 |
| random probe | 0.714 / 0.681 | 0.781 / 0.710 | 0.800 / 0.705 | 0.829 / 0.712 |
| supervised | 0.829 / 0.683 | 0.667 / 0.693 | 0.810 / 0.734 | 0.914 / 0.878 |

## Embedding quality (one image per group, seed 42; L2-normalized)

| encoder | silhouette | Davies-Bouldin | n |
|---|---|---|---|
| SSL | 0.1178 | 2.9631 | 108 |
| random-init | 0.0052 | 5.3979 | 108 |
| supervised (all) | 0.1639 | 1.9560 | 108 |

Higher silhouette and lower Davies-Bouldin = cleaner ok/defect separation. The t-SNE plot (tsne.png) is a picture, not proof; the k-NN, probe and anomaly numbers are the proof.
