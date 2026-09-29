"""Pool the casting corpus, dedupe, group near duplicates, and write the locked splits."""

import hashlib
import json
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

TARGET_PX = 96          # 128 px was 3.84 s per 128-view CPU step, too slow
SEED = 42
SPLIT_DIR = Path("splits")
HASH_CACHE = SPLIT_DIR / "hashes.npz"
CLASSES = {"ok_front": 0, "def_front": 1}   # 0 = normal, 1 = defect
# (hash key, max Hamming distance). The first two are printed for the record only; group_id uses the
# min over 8 orientations, because 180-degree rotated and mirrored re-shots of a casting are 30-50 bits
# apart at identity. Mixed-label groups are allowed: over-grouping cannot leak.
HASH_SETTINGS = {"64-bit <= 10": (8, 10), "256-bit <= 24": (16, 24), "256-bit 8-orient <= 24": ("d8", 24)}
GROUP_SETTING = "256-bit 8-orient <= 24"
# 8 orientations (dihedral group): (quarter turns counter-clockwise, mirror); index 0 is the identity
ORIENTATIONS = [(k, m) for m in (False, True) for k in range(4)]
ORIENTATION_NAMES = [f"rot{90 * k}" + ("+mirror" if m else "") for k, m in ORIENTATIONS]


def read_data_root():
    """Return the dataset root from data_path.txt (quotes and whitespace stripped)."""
    return Path(Path("data_path.txt").read_text(encoding="utf-8").strip().strip('"'))


def md5_of(path):
    """Return the MD5 hex digest of a file's bytes."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def load_corpus():
    """List the 1,300 casting_512x512 originals sorted by path, add MD5, drop exact duplicates."""
    corpus = read_data_root() / "casting_512x512" / "casting_512x512"
    rows = [{"path": str(p), "label": label}
            for folder, label in CLASSES.items()
            for p in (corpus / folder).glob("*.jpeg")]
    df = pd.DataFrame(rows).sort_values("path", ignore_index=True)
    df["md5"] = [md5_of(p) for p in df["path"]]
    n_before = len(df)
    df = df.drop_duplicates("md5", keep="first").reset_index(drop=True)
    print(f"corpus: {n_before} images, {n_before - len(df)} exact MD5 duplicates dropped, {len(df)} kept")
    return df


def orient(arr, k, mirror):
    """Rotate a 2-D array by k * 90 degrees (counter-clockwise), then mirror left-right if asked."""
    a = np.rot90(arr, k)
    return np.fliplr(a) if mirror else a


def unorient(arr, k, mirror):
    """Inverse of orient(): undo the mirror, then rotate back."""
    a = np.fliplr(arr) if mirror else arr
    return np.rot90(a, -k)


def d8_hashes(gray):
    """256-bit pHash of all 8 orientations of a grayscale PIL image, in ORIENTATIONS order."""
    arr = np.asarray(gray)
    return [imagehash.phash(Image.fromarray(np.ascontiguousarray(orient(arr, k, m))), hash_size=16).hash.flatten()
            for k, m in ORIENTATIONS]


def compute_hashes(paths):
    """Return {8: 64-bit, 16: 256-bit, "d8": 256-bit x 8 orientations}, cached in splits/hashes.npz."""
    paths = list(paths)
    if HASH_CACHE.exists():
        cache = np.load(HASH_CACHE, allow_pickle=False)
        if list(cache["paths"]) == paths and "h256_d8" in cache.files:
            return {8: cache["h64"], 16: cache["h256"], "d8": cache["h256_d8"]}
    bits = {8: [], 16: [], "d8": []}
    for p in paths:
        with Image.open(p) as img:
            gray = img.convert("L")
            for size in (8, 16):
                bits[size].append(imagehash.phash(gray, hash_size=size).hash.flatten())
            bits["d8"].append(d8_hashes(gray))
    bits = {key: np.array(b, dtype=bool) for key, b in bits.items()}   # d8: (n, 8, 256)
    SPLIT_DIR.mkdir(exist_ok=True)
    np.savez(HASH_CACHE, paths=np.array(paths), h64=bits[8], h256=bits[16], h256_d8=bits["d8"])
    return bits


def hamming_matrix(bits):
    """All-pairs Hamming distance for a bool bit matrix (n, bits)."""
    b = bits.astype(np.int32)
    return b @ (1 - b).T + (1 - b) @ b.T


def components(dist, max_dist):
    """Connected components of pairs with distance <= max_dist; return group label per image."""
    adj = dist <= max_dist
    np.fill_diagonal(adj, False)
    _, groups = connected_components(csr_matrix(adj), directed=False)
    return groups


def group_stats(groups, labels):
    """Images linked, multi-image groups, largest group, and groups mixing ok and defect."""
    g = pd.DataFrame({"g": groups, "y": labels}).groupby("g")["y"].agg(["size", "nunique"])
    multi = g[g["size"] > 1]
    return {"images linked": int(multi["size"].sum()),
            "multi-image groups": len(multi),
            "largest group": int(g["size"].max()),
            "groups mixing ok+defect": int((g["nunique"] > 1).sum())}


def cross_hamming(a, b):
    """Hamming distances between two bool bit matrices (n_a, bits) and (n_b, bits)."""
    a, b = a.astype(np.int32), b.astype(np.int32)
    return a @ (1 - b).T + (1 - a) @ b.T


def d8_distance(d8_a, d8_b):
    """Min Hamming over 8 orientations of a vs identity b, and of b vs identity a (symmetric)."""
    best = np.min([cross_hamming(d8_a[:, o], d8_b[:, 0]) for o in range(d8_a.shape[1])], axis=0)
    back = np.min([cross_hamming(d8_a[:, 0], d8_b[:, o]) for o in range(d8_b.shape[1])], axis=0)
    return np.minimum(best, back)


def near_dup_groups(df):
    """Hash every image, print the grouping table, return df with group_id from the 8-orientation distance."""
    bits = compute_hashes(df["path"])
    table, groups = {}, {}
    for name, (size, max_dist) in HASH_SETTINGS.items():
        dist = d8_distance(bits["d8"], bits["d8"]) if size == "d8" else hamming_matrix(bits[size])
        groups[name] = components(dist, max_dist)
        table[name] = group_stats(groups[name], df["label"].to_numpy())
    print(f"\nnear-duplicate groups ({len(df)} images)")
    print(pd.DataFrame(table).to_string())
    Path("results").mkdir(exist_ok=True)   # saved for report.ipynb, which reads only results/
    Path("results/grouping_stats.json").write_text(json.dumps(
        {"n_images": len(df), "group_setting": GROUP_SETTING, "settings": table}, indent=2))
    df = df.copy()
    df["group_id"] = groups[GROUP_SETTING]
    return df


def first_fold(df, n_splits):
    """Return (rest, fold 0) from StratifiedGroupKFold on label with group_id groups."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rest_idx, fold_idx = next(sgkf.split(df, df["label"], df["group_id"]))
    return df.iloc[rest_idx].reset_index(drop=True), df.iloc[fold_idx].reset_index(drop=True)


def make_splits(force=False):
    """Lock test (fold 0 of 5), then val (fold 0 of 6 on the rest); write train/val/test/pretrain CSVs."""
    names = ("train", "val", "test", "pretrain")
    if not force and all((SPLIT_DIR / f"{n}.csv").exists() for n in names):
        # Splits are made once; later runs leave the saved files alone (data.py only ever writes test.csv).
        print("splits already on disk and locked; not re-splitting (use --force to rebuild)")
        return None
    df = near_dup_groups(load_corpus())   # dedup + grouping happen before any split
    rest, test = first_fold(df, 5)
    train, val = first_fold(rest, 6)
    SPLIT_DIR.mkdir(exist_ok=True)
    cols = ["path", "label", "group_id", "md5"]
    splits = {"train": train, "val": val, "test": test}
    for name, part in splits.items():
        part[cols].to_csv(SPLIT_DIR / f"{name}.csv", index=False)
    train[["path"]].to_csv(SPLIT_DIR / "pretrain.csv", index=False)   # SimCLR list = train pool only
    print_split_sizes(splits)
    return splits


def print_split_sizes(splits):
    """Print images, ok, defect, and group counts per split."""
    rows = {name: {"images": len(p), "ok": int((p["label"] == 0).sum()),
                   "defect": int((p["label"] == 1).sum()), "groups": p["group_id"].nunique()}
            for name, p in splits.items()}
    print("\nsplits (seed 42)")
    print(pd.DataFrame(rows).T.to_string())


def load_split(name, allow_test=False, columns=None):
    """Load one split CSV (optionally only some columns); test.csv needs allow_test=True."""
    if name == "test" and not allow_test:
        raise PermissionError("test.csv is locked: only check_splits.py, eval.py and contaminated.py may open it")
    return pd.read_csv(SPLIT_DIR / f"{name}.csv", usecols=columns)


_IMAGE_CACHE = {}   # split name -> (uint8 images (n, px, px), labels (n,))


def resize_paths(paths, px=TARGET_PX):
    """Resize each image once to px grayscale; return a uint8 array (n, px, px)."""
    out = np.empty((len(paths), px, px), dtype=np.uint8)
    for i, p in enumerate(paths):
        with Image.open(p) as img:
            out[i] = np.asarray(img.convert("L").resize((px, px), Image.BILINEAR))  # PIL antialiases on downscale
    return out


def load_images(split, allow_test=False):
    """Return (uint8 images (n, 96, 96), labels) for a split, cached in memory after the first call."""
    if split not in _IMAGE_CACHE:
        df = load_split(split, allow_test=allow_test)
        _IMAGE_CACHE[split] = (resize_paths(df["path"]), df["label"].to_numpy())
    return _IMAGE_CACHE[split]


if __name__ == "__main__":
    import sys
    if "--stats" in sys.argv:   # grouping table only (hash cache), writes results/grouping_stats.json; no re-split
        near_dup_groups(load_corpus())
    else:
        make_splits(force="--force" in sys.argv)
