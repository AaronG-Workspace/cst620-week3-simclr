"""Write path-sanitized copies of split and strata files to sanitized/ for sharing; originals are untouched.

The local dataset root (from data_path.txt) is replaced by <DATA_ROOT>. The originals stay hashed in
results/frozen_manifest.json; sanitized/path_sanitization.json records each original's SHA-256 next to its
sanitized copy's, so the only change is the path prefix.
"""

import hashlib
import json
from pathlib import Path

import data

OUT = Path("sanitized")
PLACEHOLDER = "<DATA_ROOT>"
FILES = ["splits/train.csv", "splits/val.csv", "splits/test.csv", "splits/pretrain.csv",
         "splits/pretrain_contaminated.csv", "splits/budgets.json",
         "splits/pre_fix/train.csv", "splits/pre_fix/val.csv", "splits/pre_fix/test.csv",
         "splits/pre_fix/pretrain.csv", "splits/pre_fix/budgets.json",
         "results/test_strata.json", "results/contaminated/test_strata.json", "results/dry_run/val_strata.json"]


def sha256_bytes(b):
    """SHA-256 hex digest of bytes."""
    return hashlib.sha256(b).hexdigest()


def sanitize_text(text, root):
    """Replace the data root in plain form (CSV) and JSON-escaped form (backslashes doubled)."""
    for form in (root, root.replace("\\", "\\\\")):
        text = text.replace(form, PLACEHOLDER)
    return text


def main():
    """Sanitize every listed file, refuse if the local user folder survives anywhere, write the record."""
    root = str(data.read_data_root())
    user_dir = str(Path.home())                  # e.g. C:\Users\<name>; must not survive in any copy
    record = {"placeholder": PLACEHOLDER, "note": "only the local dataset root was replaced", "files": {}}
    for rel in FILES:
        src = Path(rel)
        if not src.exists():
            continue
        raw = src.read_bytes()
        clean = sanitize_text(raw.decode("utf-8"), root)
        for form in (user_dir, user_dir.replace("\\", "\\\\"), Path.home().name):
            if form in clean:
                raise SystemExit(f"REFUSING: {rel} still contains the local user folder after sanitizing")
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(clean, encoding="utf-8", newline="")
        record["files"][rel] = {"original_sha256": sha256_bytes(raw),
                                "sanitized_sha256": sha256_bytes(clean.encode("utf-8")),
                                "paths_replaced": raw.decode("utf-8").count(root) + raw.decode("utf-8").count(root.replace("\\", "\\\\"))}
    (OUT / "path_sanitization.json").write_text(json.dumps(record, indent=2))
    print(f"sanitized {len(record['files'])} files -> {OUT}/ (record: {OUT}/path_sanitization.json)")
    for rel, r in record["files"].items():
        print(f"  {rel:<42} {r['paths_replaced']:>5} paths  orig {r['original_sha256'][:12]}  sanitized {r['sanitized_sha256'][:12]}")


if __name__ == "__main__":
    main()
