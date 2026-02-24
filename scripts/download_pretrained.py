"""
scripts/download_pretrained.py
───────────────────────────────────────────────────────────────────────────────
Download PaddleOCR PP-OCRv4 English recognition training-set checkpoints into
the ``pretrained/`` directory.

Usage
-----
    python scripts/download_pretrained.py              # download mobile (default)
    python scripts/download_pretrained.py --model server   # download server
    python scripts/download_pretrained.py --list           # list available models

After running this script, start fine-tuning with:

    python train.py --pretrained-model pretrained/en_PP-OCRv4_rec_train/best_accuracy

Why use training checkpoints (not inference models)?
-----------------------------------------------------
The inference package (.pdmodel / .pdiparams) is a static graph with fixed
dimensions and cannot be fine-tuned.  The training checkpoint
(best_accuracy.pdparams) contains the full parameter tensors that PaddleOCR's
``tools/train.py`` loads with ``load_model()``, allowing weight updates.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

# ── Model registry ────────────────────────────────────────────────────────────

MODELS = {
    "mobile": {
        "url": (
            "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/"
            "en_PP-OCRv4_rec_train.tar"
        ),
        "dest_dir": "en_PP-OCRv4_rec_train",
        "description": "PP-OCRv4 English mobile (MobileNetV1Enhance + SVTR, ~11 MB)",
    },
    "server": {
        "url": (
            "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/"
            "en_PP-OCRv4_rec_server_train.tar"
        ),
        "dest_dir": "en_PP-OCRv4_rec_server_train",
        "description": "PP-OCRv4 English server (ResNet45 + SVTR, higher accuracy, ~90 MB)",
    },
    "ch_mobile": {
        "url": (
            "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/"
            "ch_PP-OCRv4_rec_train.tar"
        ),
        "dest_dir": "ch_PP-OCRv4_rec_train",
        "description": "PP-OCRv4 Chinese mobile (use only if your data has Chinese chars)",
    },
}

# Project root: two directories up from this script (scripts/ → project root)
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
_PRETRAINED_DIR = _PROJECT_ROOT / "pretrained"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        mb = downloaded / 1_048_576
        total_mb = total_size / 1_048_576
        bar = "#" * int(pct / 2)
        sys.stdout.write(f"\r  [{bar:<50}] {pct:5.1f}%  {mb:.1f}/{total_mb:.1f} MB")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\r  Downloaded {downloaded / 1_048_576:.1f} MB")
        sys.stdout.flush()


def _download(url: str, dest: Path) -> None:
    print(f"  Downloading: {url}")
    urllib.request.urlretrieve(url, str(dest), reporthook=_progress_hook)
    print()  # newline after progress bar


def _extract_tar(tar_path: Path, target_dir: Path) -> None:
    print(f"  Extracting {tar_path.name} …")
    with tarfile.open(str(tar_path)) as tar:
        tar.extractall(path=str(target_dir))


# ── Main ──────────────────────────────────────────────────────────────────────

def download_model(name: str) -> Path:
    if name not in MODELS:
        raise ValueError(
            f"Unknown model '{name}'.  Available: {list(MODELS.keys())}"
        )

    info = MODELS[name]
    url: str = info["url"]
    dest_dir = _PRETRAINED_DIR / info["dest_dir"]

    # Already downloaded?
    checkpoint = dest_dir / "best_accuracy.pdparams"
    if checkpoint.exists():
        print(f"  ✓ Already downloaded: {dest_dir}")
        return dest_dir

    _PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    tar_dest = _PRETRAINED_DIR / Path(url).name
    try:
        _download(url, tar_dest)
        _extract_tar(tar_dest, _PRETRAINED_DIR)
    finally:
        if tar_dest.exists():
            tar_dest.unlink()

    if not checkpoint.exists():
        raise RuntimeError(
            f"Expected {checkpoint} after extraction but it was not found.  "
            "The archive may have a different internal layout — check the "
            "extracted contents manually."
        )

    print(f"  ✓ Saved to: {dest_dir}")
    return dest_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download PP-OCRv4 recognition pretrained checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        default="mobile",
        choices=list(MODELS.keys()),
        help="Which pretrained model to download.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list:
        print("\nAvailable pretrained models:\n")
        for name, info in MODELS.items():
            print(f"  {name:<12}  {info['description']}")
        print(
            "\nUsage after download:\n"
            "  python train.py --pretrained-model pretrained/<dest_dir>/best_accuracy\n"
        )
        return

    print(f"\nDownloading PP-OCRv4 '{args.model}' pretrained checkpoint …\n")
    dest_dir = download_model(args.model)

    print(
        f"\nSuccess!  Use this model for fine-tuning:\n"
        f"  python train.py \\\n"
        f"      --pretrained-model {dest_dir}/best_accuracy \\\n"
        f"      --epochs 50 --lr 5e-5\n"
    )


if __name__ == "__main__":
    main()
