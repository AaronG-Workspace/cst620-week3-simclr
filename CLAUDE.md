# CST 620 Week 3 — Self-supervised representation audit (SimCLR) for industrial defect detection

- The assignment brief is in ASSIGNMENT.md. Read it first. Where it conflicts with this file, the brief wins.
- This project produces the pipeline, results/, and an executed report.ipynb (embedding plots + linear probe results). The graded video is a separate slide walkthrough, so every figure must be slide ready.
- Environment: Windows, PowerShell, CPU only. Every single run must finish in under 8 minutes. Time one epoch before any long run. DataLoader num_workers=0.
- The 8-minute limit applies to runs started from VS Code (one command per run). Stages inside run.py may take up to 10 minutes each (e.g. the supervised "all" budget, 20 epochs, 600 s guard).
- Runs longer than 8 minutes (including python run.py) are started by the user in the second terminal, not by Claude.
- Print ASCII only. Week 2 crashed printing a Greek letter on the Windows console, so write "tau", never the symbol.
- Keep replies brief: one compact table or a few lines per step. No Insight sections, no long explanations.

## Data
- data_path.txt holds the dataset root (Kaggle ravirajsinh45/real-life-industrial-dataset-of-casting-product). Label 0 = ok_front (normal), 1 = def_front (defect).
- Corpus = casting_512x512/casting_512x512/{ok_front,def_front}, the 1,300 un-augmented originals. Resize once to 96 px grayscale (128 px took 3.84 s per 128-view CPU step, too slow for enough epochs), cache in memory, replicate to 3 channels.
- casting_data/ (the 300 px set) is EXCLUDED. Its images are pre-augmented copies of the originals and cannot be traced back to a source image, so they cannot be proven disjoint from test. audit_source.py measures how much that set's own train/test split leaks (MD5 exact matches, and 256-bit pHash Hamming <= 24) and writes results/source_audit.json. No other file reads casting_data/.

## Leakage rules (the point of this week)
- Drop exact duplicates by MD5 before anything else.
- Near duplicates: imagehash.phash(img, hash_size=16) (256-bit), Hamming <= 24. Do NOT use the 64-bit hash at <= 10 here: every casting looks alike at that resolution, so it chains almost the whole dataset into one group and cannot see a defect. Print group stats for both settings so the comparison is on record.
- A near-duplicate pair is the same casting photographed again. The pair distance is the minimum 256-bit Hamming over all 8 orientations (rotate 0/90/180/270, each with and without a mirror), because the dataset holds 180-degree rotated and mirrored re-shots of the same casting that sit 30-50 bits apart at identity; the identity-only check left 44 val and 101 test images with a rotated twin in train. Connected components of pairs <= 24 = group_id (physical object). Mixed-label groups are allowed, because over-grouping cannot leak. Keep these images, but a group never spans two splits.
- Lock the test split first: StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42), fold 0 = test. Then StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42) on the rest, fold 0 = val. The rest is the train pool. Split once and save splits/{train,val,test}.csv with columns path,label,group_id,md5.
- SimCLR pretraining list = train pool only. Val is held out from pretraining because it tunes tau, k, and probe settings.
- check_splits.py fails loudly unless: no path, MD5, or group_id appears in two splits; zero cross-split pairs at 256-bit Hamming <= 24 under any of the 8 orientations; the pretraining list is a subset of train.csv. Print the counts.
- Training and tuning code never opens test.csv. The clean test set is scored once, by eval.py, after every choice is frozen.
- Test strata: frozen_choices.json stores only the rule. near = test image whose nearest train image by random-init (seed 42) feature cosine is within 8-orientation pHash distance <= 40; clean = the rest. eval.py computes the lists from test images and train features before it reads any test label, saves them to results/test_strata.json, and reports every method on both strata.
- Freeze before test: python baselines.py --freeze writes results/frozen_manifest.json (SHA-256 of weights/ssl_encoder.pt, every supervised weight file eval.py scores, splits/budgets.json, results/frozen_choices.json, and the split CSVs, plus a timestamp). eval.py must recompute every hash and refuse to run if any file is missing or any hash changes. Never re-freeze after test has been scored.
- test.csv access: data.py writes it. check_splits.py reads only its path, md5, and group_id columns. eval.py and contaminated.py are the only files that load test images. No training or tuning file opens it.
- Any k-NN or silhouette computed within one split must exclude same-group neighbors: k-NN is leave-one-group-out (neighbors with the query's group_id are masked), and silhouette uses one image per group (sampled with seed 42). Twins stay in the same split by design, so plain leave-one-out would let them vote for each other.
- Never compute statistics from test images. Fixed normalization (mean 0.5, std 0.5). Probe feature scaling is fit on the labeled train subset only.
- No pretrained weights anywhere: torchvision resnet18(weights=None). An ImageNet checkpoint is a pretraining corpus nobody can audit.

## SimCLR
- ResNet-18 encoder with fc = Identity (512-d). 2-layer MLP projection head (512 -> 512 -> 128) for pretraining only. Evaluate the ENCODER output, never the projection head output.
- NT-Xent on L2-normalized projections, cosine similarity, temperature from config. Batch 64, so 126 negatives per anchor (2N - 2). Log batch size and negative count in results.
- Augmentations must keep the defect in view. Castings are round and shot from the top, so use RandomRotation(0-360), horizontal and vertical flips, RandomResizedCrop(96, scale=(0.9, 1.0), ratio=(1.0, 1.0)) so castings stay round, brightness and contrast jitter only (grayscale, so no hue or saturation), and no blur. Save results/aug_check.png: 4 defect images x 6 augmented views, so defect survival can be checked by eye.
- Seed the augmentation RNG with its own torch.Generator, separate from the model seed, so runs are reproducible.
- Log every epoch: loss, val k-NN accuracy (k=5, cosine, val labels only), and embedding std across dimensions (collapse check). Save results/pretrain_history.json and results/pretrain_curve.png.
- Temperature: short runs at tau 0.1, 0.2, 0.5, pick by val k-NN, then one full run with the winner. Adam, lr 1e-3 unless config says otherwise.

## Evaluation (frozen encoders, same clean test split for everything)
- Label budgets per class: 10, 25, 50 (3 draws each, seeds 0 to 2) plus all train labels (1 draw). Every method at a budget uses the exact same labeled image IDs. Save them to splits/budgets.json.
- Methods at every budget:
  - SSL encoder + k-NN (cosine, k from {1, 5, 10, 20} chosen on val). This is the headline number.
  - SSL encoder + linear probe (LogisticRegression, C chosen on val), reported alongside k-NN.
  - Random-init ResNet-18 + the same k-NN and probe (the floor).
  - Supervised ResNet-18 trained from scratch on the same labeled images (same resolution, rotations and flips, early stopping on val).
- Report mean +/- std across draws. Flag when the probe beats k-NN by more than 10 points, and when k-NN beats the probe.
- Anomaly detection: score = mean cosine distance to the k nearest embeddings of the labeled ok images in the budget. Report AUROC with a 95% bootstrap CI. Pick the threshold on val, then report on test: missed defects (false accepts), false rejects, and per-class recall, not only overall accuracy.
- Embedding quality on test: silhouette and Davies-Bouldin for SSL vs random-init vs supervised encoders. results/tsne.png with 3 panels colored by label. A t-SNE plot is a picture, not proof; the probe and k-NN are the proof.
- eval.py writes results/test_metrics.json and results/comparison.md.
- Contaminated control, only after the clean test scores are written: contaminated.py pretrains again with the same config on train pool + test images, eval.py scores it the same way, and the gap (contaminated minus clean) is the leakage estimate. contaminated.py is the only other file allowed to read test.csv. Label it CONTAMINATED everywhere.

## Project
- Files: data.py, audit_source.py, check_splits.py, simclr.py, train_ssl.py, baselines.py, eval.py, contaminated.py, run.py, report.ipynb. Short commented functions. Outputs to results/ and weights/.
- report.ipynb only reads results/ and weights/ (no training, no test scoring). Execute it with: python -m jupyter nbconvert --to notebook --execute --inplace report.ipynb
- Figures: PNG, 150 dpi, large fonts, one idea per figure.
