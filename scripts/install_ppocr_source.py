"""
scripts/install_ppocr_source.py
───────────────────────────────────────────────────────────────────────────────
Download the ``ppocr/`` training module from the PaddleOCR 2.7 GitHub
repository using a Git sparse-checkout so only the required directory is
fetched (not the full multi-GB repository).

Why is this needed?
───────────────────
The ``paddleocr`` pip package (2.7.x) ships only a partial ``ppocr`` module
(data, postprocess, utils) sufficient for inference.  The training pipeline
in ``tools/train.py`` additionally needs:

    ppocr.modeling   – model architectures (SVTR_LCNet, etc.)
    ppocr.losses     – CTCLoss, etc.
    ppocr.optimizer  – Adam + Cosine LR builder
    ppocr.metrics    – RecMetric, etc.

These are only included in the PaddleOCR source repository.  This script
clones just the ``ppocr/`` subtree (sparse clone, ~4 MB) and places it at
the project root so Python can import it directly.

Usage
─────
    python scripts/install_ppocr_source.py          # installs if not present
    python scripts/install_ppocr_source.py --force  # re-downloads even if present
    python scripts/install_ppocr_source.py --check  # verify existing install
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
PPOCR_DEST   = PROJECT_ROOT / "ppocr"
CLONE_URL    = "https://github.com/PaddlePaddle/PaddleOCR.git"
BRANCH       = "release/2.7"

# These sub-modules must exist for training to work.
REQUIRED_SUBMODULES = ["modeling", "losses", "optimizer", "metrics", "data"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_existing() -> bool:
    """Return True if the ppocr install looks complete."""
    if not PPOCR_DEST.exists():
        return False
    missing = [m for m in REQUIRED_SUBMODULES if not (PPOCR_DEST / m).exists()]
    if missing:
        print(f"  [warn] ppocr/ exists but is incomplete — missing: {missing}")
        return False
    return True


def _run(cmd: list[str], cwd: str | None = None) -> None:
    """Run a shell command, raising SystemExit on failure."""
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed: {' '.join(cmd)}")
        sys.exit(result.returncode)


def download() -> None:
    """Sparse-clone only the ppocr/ subtree from PaddleOCR 2.7."""
    tmp_dir = PROJECT_ROOT / "_paddleocr_src_tmp"

    if tmp_dir.exists():
        shutil.rmtree(str(tmp_dir))

    print(f"  Cloning PaddleOCR {BRANCH} (sparse — ppocr/ only) …")

    # Step 1: create a sparse clone (no file content yet, just git objects)
    _run([
        "git", "clone",
        "--filter=blob:none",   # lazy blob download
        "--sparse",             # only root files until sparse-checkout set
        "--branch", BRANCH,
        "--depth", "1",
        "--no-tags",
        CLONE_URL,
        str(tmp_dir),
    ])

    # Step 2: restrict the working tree to ppocr/ only
    _run(
        ["git", "sparse-checkout", "set", "ppocr"],
        cwd=str(tmp_dir),
    )

    # Step 3: copy ppocr/ to the project root
    src = tmp_dir / "ppocr"
    if not src.exists():
        print(f"\n[ERROR] Sparse-checkout did not produce {src}")
        shutil.rmtree(str(tmp_dir))
        sys.exit(1)

    if PPOCR_DEST.exists():
        shutil.rmtree(str(PPOCR_DEST))
    shutil.copytree(str(src), str(PPOCR_DEST))

    # Step 4: clean up temp directory
    shutil.rmtree(str(tmp_dir))

    print(f"  ✓ ppocr source installed → {PPOCR_DEST}")


def verify() -> None:
    """Print import verification results."""
    sys.path.insert(0, str(PROJECT_ROOT))
    errors = []
    for mod in REQUIRED_SUBMODULES:
        try:
            __import__(f"ppocr.{mod}")
            print(f"  ✓ ppocr.{mod}")
        except ImportError as e:
            print(f"  ✗ ppocr.{mod}  ({e})")
            errors.append(mod)

    if errors:
        print(f"\nFailed modules: {errors}")
        sys.exit(1)
    else:
        print("\nAll ppocr training modules importable.")


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Install ppocr training source from PaddleOCR 2.7 GitHub."
    )
    p.add_argument("--force",  action="store_true",
                   help="Re-download even if ppocr/ already exists.")
    p.add_argument("--check",  action="store_true",
                   help="Verify existing install and exit.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.check:
        print(f"\nChecking ppocr install at {PPOCR_DEST} …\n")
        verify()
        return

    if _check_existing() and not args.force:
        print(f"ppocr/ already installed at {PPOCR_DEST}")
        print("Run with --force to re-download, or --check to verify.")
        return

    print(f"\nInstalling ppocr training source → {PPOCR_DEST}\n")
    download()
    print()
    print("Verifying imports …")
    verify()
    print()
    print("Done.  You can now run:  python train.py")


if __name__ == "__main__":
    main()
