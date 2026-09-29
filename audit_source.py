"""Measure leakage inside the casting_data train/test split; the only file that reads casting_data/."""

import json
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

import data

PHASH_SIZE = 16          # 256-bit pHash
PHASH_MAX_DIST = 24
OUT_PATH = Path("results/source_audit.json")


def list_source_split():
    """List casting_data/casting_data/{train,test} images as a DataFrame (split, path, label, md5)."""
    root = data.read_data_root() / "casting_data" / "casting_data"
    rows = [{"split": split, "path": str(p), "label": label}
            for split in ("train", "test")
            for folder, label in data.CLASSES.items()
            for p in sorted((root / split / folder).glob("*.jpeg"))]
    df = pd.DataFrame(rows)
    df["md5"] = [data.md5_of(p) for p in df["path"]]
    return df


def phash_bits(paths):
    """256-bit pHash bit matrix (n, 256) for grayscale images."""
    out = []
    for p in paths:
        with Image.open(p) as img:
            out.append(imagehash.phash(img.convert("L"), hash_size=PHASH_SIZE).hash.flatten())
    return np.array(out, dtype=bool)


def min_hamming(query, ref):
    """Distance from each query hash to its nearest ref hash."""
    q, r = query.astype(np.int32), ref.astype(np.int32)
    return (q @ (1 - r).T + (1 - q) @ r.T).min(axis=1)


def row(name, hits, total):
    """One table row: count, total, percent."""
    return {"check": name, "count": int(hits), "total": int(total), "pct": round(100 * hits / total, 1)}


def run_audit():
    """Compute the three leakage counts for the provided split and against the 1,300 originals."""
    src = list_source_split()
    tr, te = src[src["split"] == "train"], src[src["split"] == "test"]
    bits = phash_bits(src["path"])
    bits_tr, bits_te = bits[(src["split"] == "train").to_numpy()], bits[(src["split"] == "test").to_numpy()]

    orig = data.load_corpus()
    orig_bits = data.compute_hashes(orig["path"])[PHASH_SIZE]   # reuses splits/hashes.npz

    md5_hits = te["md5"].isin(set(tr["md5"])).sum()
    near_hits = (min_hamming(bits_te, bits_tr) <= PHASH_MAX_DIST).sum()
    orig_dist = min_hamming(bits, orig_bits)
    return [
        row("provided test: MD5 match in provided train", md5_hits, len(te)),
        row("provided test: 256-bit neighbor <= 24 in provided train", near_hits, len(te)),
        row(f"300 px (all): 256-bit neighbor <= 24 in {len(orig):,} originals", (orig_dist <= PHASH_MAX_DIST).sum(), len(src)),
        row("  of which provided train", (orig_dist[: len(tr)] <= PHASH_MAX_DIST).sum(), len(tr)),
        row("  of which provided test", (orig_dist[len(tr):] <= PHASH_MAX_DIST).sum(), len(te)),
    ]


def write_audit(rows):
    """Write the source-split leakage report to results/source_audit.json."""
    OUT_PATH.parent.mkdir(exist_ok=True)
    report = {"hash": "imagehash.phash hash_size=16 (256-bit)", "max_hamming": PHASH_MAX_DIST, "checks": rows}
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="ascii")


def main():
    """Run the source audit, print one ASCII table, write JSON."""
    rows = run_audit()
    write_audit(rows)
    print("\nsource audit: casting_data/casting_data (EXCLUDED from all training and eval)")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
