# SimCLR representation audit for casting defect detection

Can self-supervised contrastive pretraining (SimCLR) replace expensive manual labels for visual defect inspection?
This project pretrains a ResNet-18 with SimCLR on unlabeled images, then compares it with a random-init encoder and a
supervised model trained from scratch, at 10, 25 and 50 labels per class and with all labels. Every comparison uses
the same labeled images and the same untouched test set.

- **Data:** [Kaggle: real-life industrial dataset of casting product](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product),
  the 1,300 un-augmented 512 px originals (`ok_front` = normal, `def_front` = defect). Images are not included here.
- **Compute:** CPU only. 96 px grayscale, batch 64 (126 negatives per anchor), 10 pretraining epochs of about 42 s each.
- **Full write-up:** [`report.ipynb`](report.ipynb) (executed) or [`report.html`](report.html).

## Results (test, 260 images, scored once)

Accuracy, mean ± std over 3 label draws ("all" is one draw). Always predicting "defect" scores 60.0%.

| method | 10 / class | 25 / class | 50 / class | all labels |
|---|---|---|---|---|
| SSL encoder + linear probe | 0.621 ± 0.002 | 0.676 ± 0.056 | **0.738 ± 0.042** | **0.912** |
| SSL encoder + k-NN | **0.656 ± 0.025** | **0.681 ± 0.023** | 0.638 ± 0.044 | 0.758 |
| Random-init encoder + linear probe | 0.619 ± 0.038 | 0.658 ± 0.028 | 0.676 ± 0.028 | 0.769 |
| Random-init encoder + k-NN | 0.626 ± 0.021 | 0.655 ± 0.049 | 0.662 ± 0.050 | 0.877 |
| Supervised ResNet-18 from scratch | 0.636 ± 0.024 | 0.564 ± 0.036 | 0.668 ± 0.027 | 0.835 |

- **With all labels, the SSL linear probe is the best method** (0.912). It scores 0.914 on the 58 test images
  that look most like a train image and 0.911 on the other 202, so the result doesn't rest on near-duplicates.
- **With 10 to 50 labels per class, SSL features give only a small edge.** SSL leads at every small budget, but
  mostly by less than the spread between draws, and recall on good parts stays low (0.37 to 0.62).
- **Anomaly detection** (cosine distance to labeled good parts): SSL features give AUROC 0.71 to 0.72 at every
  budget, with 95% group-bootstrap intervals of about [0.55, 0.87]. At the threshold that catches 95% of defects,
  they still reject 76% to 89% of good parts, so this is **not a usable inspection gate** at this compute budget.
- **Embedding quality** (one image per physical part): silhouette 0.208 for SSL, 0.038 for random-init and 0.243
  for supervised. SSL learned real structure, but less than supervised training does.
- **Contamination control:** pretraining again with the test images included changed SSL test accuracy and AUROC
  by about 0 to 5 points, mostly downward, so the clean results aren't propped up by leakage.

**Verdict:** at this lab-sized compute budget, SimCLR pretraining pays off when many labels are available
(+14 points over the linear probe on random features, +8 over supervised from scratch). It does not yet replace
labeling in the low-label regime, where its gains sit inside the noise. Longer pretraining, larger batches or
higher resolution, and a larger test set, would be needed to change that verdict.

## Evaluation integrity

The dataset contains re-shots of the same physical casting, many of them rotated 180 degrees or mirrored. Leakage
controls:

1. **Deduplicate and group before splitting.** Exact duplicates are dropped by MD5. Near duplicates are linked by
   256-bit pHash with Hamming distance ≤ 24 under **all 8 orientations**. Identity-only hashing had left 44 val and
   101 test images with a rotated twin in train (see `results/pre_fix/`). Linked images form a `group_id` (one
   physical part), and a group never spans two splits.
2. **Lock the test split first:** StratifiedGroupKFold, seed 42. SimCLR pretrains on the train pool only.
   [`check_splits.py`](check_splits.py) proves the pretraining list is disjoint from test by path, MD5, group and
   8-orientation pHash, and it fails loudly on a contaminated list.
3. **Tune on val only.** Val k-NN is leave-one-group-out, so twins inside val can't vote for each other.
4. **Freeze before test.** [`results/frozen_manifest.json`](results/frozen_manifest.json) holds SHA-256 hashes of
   every weight file, choice and split. [`eval.py`](eval.py) refuses to run if any hash changes, computes the
   near/clean test strata before reading a single test label, scores test once, then locks.
5. **Measure leakage directly** with a CONTAMINATED pretraining run ([`contaminated.py`](contaminated.py)); results
   are in `results/contaminated/`.

The provided 300 px split that ships with the dataset leaks badly (9% of its test images are exact copies of train
images and 46% are near copies; `results/source_audit.json`), so it is excluded.

## Layout

| file | role |
|---|---|
| `data.py` | pool, dedupe, 8-orientation grouping, locked group split |
| `audit_source.py` | leakage audit of the dataset's own provided split |
| `check_splits.py` | disjointness proofs (split and pretraining list vs test) |
| `simclr.py` | encoder, projection head, defect-preserving augmentations, NT-Xent |
| `train_ssl.py` | temperature sweep and pretraining with a leave-one-group-out val monitor |
| `baselines.py` | label budgets, supervised baseline and learning-rate search, val tuning, freeze |
| `eval.py` | hash-verified single test run, strata, bootstrap CIs, embedding quality |
| `contaminated.py` | CONTAMINATED leakage control |
| `run.py` | runs every stage in order, one process each |
| `report.ipynb` | reads only `results/` and `weights/` |
| `results/` | metrics, figures, frozen manifest and choices |
| `sanitized/` | split and strata files with local paths replaced by `<DATA_ROOT>` |

## Reproducing

Needs Python 3 with PyTorch and torchvision (CPU), scikit-learn, pandas, imagehash, Pillow, matplotlib and nbconvert.

1. Download the dataset and put its root folder path (the one containing `casting_512x512/`) in `data_path.txt`.
2. Restore the exact splits by copying `sanitized/splits/` to `splits/` with `<DATA_ROOT>` replaced by that path.
   `sanitized/path_sanitization.json` lists the SHA-256 of each original file. Alternatively,
   `python data.py --force` rebuilds the splits from scratch.
3. Run `python run.py` (about 50 minutes on CPU). `python eval.py --dry-run` rehearses evaluation on val;
   `python eval.py` scores test once.

Model weights (1.6 GB) are not included. Their SHA-256 hashes are in `results/frozen_manifest.json`.

## Limitations

Small evaluation sets (val 174, test 260; AUROC intervals are about ±0.15). Pretraining is short by design of the
compute budget. Val was used for many choices, including widening the anomaly-score grid after seeing val.
8-orientation hashing cannot catch parts re-shot at arbitrary angles. Thresholds set on val did not transfer to
test. Castings stand in for wafer inspection. Details are in section 8 of the report.
