"""
train.py – Entry-point script for PaddleOCR recognition fine-tuning.

Usage
-----
# Default: processes all built-in sample sources
python train.py

# Custom sources (image + ground-truth pairs, comma-separated)
python train.py 

# Override training hyper-parameters
python train.py --epochs 50 --batch-size 16 --lr 5e-5 --output-dir runs/v2

# Skip export / evaluation after training
python train.py --no-export --no-eval
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train")

# ── Default source data ────────────────────────────────────────────────────────

_DEFAULT_SOURCES = [
    {
        # Original ground-truth for 1887 — 1500 rows × 25 cols (SLNO + 24 fields)
        "image": "sample input/HGF-1500-1887.gif",
        "gt": "original data/3.HGF-1500-1887_Original_Data.xlsx",
    }
]
# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune PaddleOCR recognition on domain-specific documents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data sources
    p.add_argument(
        "--images",
        default=None,
        help="Comma-separated list of source image paths (GIF / PNG / JPG).",
    )
    p.add_argument(
        "--ground-truths",
        default=None,
        help="Comma-separated list of Excel ground-truth paths (must match --images order).",
    )
    p.add_argument(
        "--pipeline-output",
        default="finetune_data",
        help="Directory where the data pipeline writes crops and label files.",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of data reserved for validation.",
    )

    # Training hyper-parameters
    p.add_argument("--output-dir", default="runs/finetune_v1",
                   help="Root directory for training artefacts.")
    p.add_argument("--epochs", type=int, default=50,
                   help="Total training epochs.  50 is a good starting point for fine-tuning.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5, dest="learning_rate",
                   help="Peak learning rate.  5e-5 is conservative for fine-tuning from "
                        "a PaddleOCR pretrained checkpoint; use 1e-4 for scratch training.")
    p.add_argument("--warmup-epochs", type=int, default=5,
                   help="Linear LR warmup before Cosine decay.  Prevents optimizer "
                        "shock on the first few epochs when fine-tuning.")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--img-height", type=int, default=32)
    p.add_argument("--img-width", type=int, default=320)
    p.add_argument("--max-text-length", type=int, default=50,
                   help="Maximum characters per crop.  Must be >= max observed label length "
                        "in your dataset (pipeline stats show max=45; default 50 adds margin).")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (use 0 on Windows to avoid spawn issues).")
    p.add_argument("--pretrained-model", default=None,
                   help="Path to a PaddleOCR checkpoint to start from.")
    p.add_argument("--no-gpu", action="store_true",
                   help="Disable GPU even when CUDA is available.")
    p.add_argument(
        "--no-grid-slicing",
        action="store_true",
        help=(
            "Disable grid-based cell slicing and fall back to the PaddleOCR "
            "detection + row-clustering approach.  Grid slicing is the default "
            "and is recommended for fixed-format scanned tables."
        ),
    )
    p.add_argument(
        "--expected-cols",
        type=int,
        default=25,
        help="Number of table columns the grid slicer expects (default: 25 = SLNO + 24 data cols).",
    )
    p.add_argument(
        "--crop-upscale",
        type=float,
        default=2.0,
        dest="crop_upscale",
        help=(
            "Scale factor applied to each saved cell-crop image before writing "
            "to disk.  Use 2.0 (default) to double the pixel dimensions with "
            "bicubic interpolation — recommended when raw row height is ~20 px.  "
            "Set to 1.0 to disable upscaling."
        ),
    )

    # Data quality
    p.add_argument(
        "--keep-empty-labels",
        action="store_true",
        help=(
            "Do NOT filter out records whose ground-truth label is an empty "
            "string.  By default empty-label crops (blank table cells) are "
            "removed because they add no text-recognition signal."
        ),
    )

    # Post-training
    p.add_argument("--no-export", action="store_true",
                   help="Skip exporting the best checkpoint to inference format.")
    p.add_argument("--no-eval", action="store_true",
                   help="Skip running evaluation after training.")

    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_sources(args: argparse.Namespace) -> list[dict]:
    """Return a list of {image, gt} dicts from CLI args or built-in defaults."""
    if args.images is None and args.ground_truths is None:
        logger.info("No --images/--ground-truths supplied; using built-in defaults.")
        return _DEFAULT_SOURCES

    if args.images is None or args.ground_truths is None:
        raise ValueError(
            "--images and --ground-truths must be provided together."
        )

    images = [p.strip() for p in args.images.split(",")]
    gts = [p.strip() for p in args.ground_truths.split(",")]

    if len(images) != len(gts):
        raise ValueError(
            f"Number of images ({len(images)}) must match "
            f"number of ground-truth files ({len(gts)})."
        )

    return [{"image": img, "gt": gt} for img, gt in zip(images, gts)]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # ------------------------------------------------------------------
    # 1.  Imports (deferred so --help is instant)
    # ------------------------------------------------------------------
    # Ensure the project root is on sys.path so the sibling modules
    # (config, dataset, pipeline, trainer) are importable when this
    # script is executed directly with `python train.py`.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))

    from config import FinetuneConfig
    from trainer import PaddleOCRVLFinetuner
    from pipeline import TrainingDataPipeline

    # ------------------------------------------------------------------
    # 2.  Build training dataset via the data pipeline
    # ------------------------------------------------------------------
    sources = _resolve_sources(args)

    logger.info("=== Step 1/4 – Building training dataset ===")
    use_grid = not args.no_grid_slicing
    logger.info(
        "Cell extraction mode: %s",
        "grid-slice (direct cell slicing)" if use_grid else "ocr-detect (PaddleOCR + clustering)",
    )
    pipeline = TrainingDataPipeline(
        output_dir=args.pipeline_output,
        use_grid_slicing=use_grid,
        expected_table_cols=args.expected_cols,
        crop_upscale=args.crop_upscale,
    )
    logger.info("  crop_upscale    : %.1f×", args.crop_upscale)

    for src in sources:
        image_path = src["image"]
        gt_path = src["gt"]

        if not Path(image_path).exists():
            logger.warning("Image not found, skipping: %s", image_path)
            continue
        if not Path(gt_path).exists():
            logger.warning("Ground-truth not found, skipping: %s", gt_path)
            continue

        pipeline.add_source(image_path=image_path, ground_truth_path=gt_path)
        logger.info("  + registered source: %s", image_path)

    train_ds, val_ds = pipeline.build(val_ratio=args.val_ratio)

    if not args.keep_empty_labels:
        removed_train = train_ds.filter_empty_labels()
        removed_val = val_ds.filter_empty_labels()
        if removed_train or removed_val:
            logger.info(
                "Filtered %d empty-label crops (train=%d, val=%d). "
                "Pass --keep-empty-labels to disable.",
                removed_train + removed_val, removed_train, removed_val,
            )

    logger.info(
        "Dataset ready – train: %d samples, val: %d samples.",
        len(train_ds),
        len(val_ds),
    )

    # ------------------------------------------------------------------
    # 3.  Configure and run the fine-tuner
    # ------------------------------------------------------------------
    logger.info("=== Step 2/4 – Configuring trainer ===")
    cfg = FinetuneConfig(
        output_dir=args.output_dir,
        pretrained_model_dir=args.pretrained_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_epoch=args.warmup_epochs,
        weight_decay=args.weight_decay,
        img_height=args.img_height,
        img_width=args.img_width,
        max_text_length=args.max_text_length,
        num_workers=args.num_workers,
        use_gpu=not args.no_gpu,
    )
    logger.info("  output_dir      : %s", cfg.output_dir)
    logger.info("  epochs          : %d", cfg.epochs)
    logger.info("  batch_size      : %d", cfg.batch_size)
    logger.info("  learning_rate   : %g", cfg.learning_rate)
    logger.info("  use_gpu         : %s", cfg.use_gpu)

    tuner = PaddleOCRVLFinetuner(cfg)

    logger.info("=== Step 3/4 – Training ===")
    best_model = tuner.run(train_ds, val_ds)
    logger.info("Best model checkpoint: %s", best_model)

    # ------------------------------------------------------------------
    # 4.  Export
    # ------------------------------------------------------------------
    if not args.no_export:
        logger.info("=== Step 4/4 – Exporting inference model ===")
        export_dir = tuner.export(best_model)
        logger.info("Exported model → %s", export_dir)
    else:
        logger.info("=== Step 4/4 – Export skipped (--no-export) ===")
        export_dir = None

    # ------------------------------------------------------------------
    # 5.  Evaluation  (optional)
    # ------------------------------------------------------------------
    if not args.no_eval and export_dir is not None:
        logger.info("=== Evaluating on validation set ===")
        metrics = tuner.evaluate(val_ds, model_dir=str(export_dir))
        logger.info("Validation metrics: %s", metrics)

    logger.info("=== Training complete ===")


if __name__ == "__main__":
    main()
