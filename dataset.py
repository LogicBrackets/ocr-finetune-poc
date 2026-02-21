import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AnnotationRecord:
    """A single training sample: the absolute path to an image and its text label."""

    image_path: str
    label: str
    # OCR engine's own confidence for this prediction – useful for weighting.
    confidence: float = 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Dataset manager
# ──────────────────────────────────────────────────────────────────────────────

class FinetuneDataset:
    """
    Manages the collection, validation, and serialisation of training data
    for PaddleOCR recognition fine-tuning.

    PaddleOCR label-file format (one sample per line)::

        <relative_or_absolute_image_path>\\t<label_text>

    Typical usage
    -------------
    >>> ds = FinetuneDataset("data/finetune")
    >>> ds.load_from_csv("annotations.csv", image_col="img", label_col="text")
    >>> train_ds, val_ds = ds.split_train_val()
    >>> train_ds.save_label_file("data/finetune/train_list.txt")
    """

    def __init__(self, data_dir: str) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._records: List[AnnotationRecord] = []

    # ── Data collection ───────────────────────────────────────────────────────

    def add_sample(
        self,
        image_path: str,
        label: str,
        confidence: float = 1.0,
    ) -> None:
        """Add one (image, label) pair.  Silently skips missing image files."""
        if not os.path.exists(image_path):
            logger.warning("Image not found – skipping: %s", image_path)
            return
        self._records.append(
            AnnotationRecord(image_path, label.strip(), float(confidence))
        )

    def load_from_csv(
        self,
        csv_path: str,
        image_col: str = "image_path",
        label_col: str = "label",
        confidence_col: Optional[str] = None,
    ) -> int:
        """
        Load annotations from a CSV file.

        Parameters
        ----------
        csv_path:       path to the CSV file
        image_col:      column that contains the absolute image path
        label_col:      column that contains the expected text label
        confidence_col: optional column with per-sample OCR confidence

        Returns the number of records added.
        """
        import pandas as pd

        df = pd.read_csv(csv_path)
        required = {image_col, label_col}
        if not required.issubset(df.columns):
            raise ValueError(
                f"CSV is missing required columns {required}. "
                f"Found: {list(df.columns)}"
            )

        before = len(self._records)
        for _, row in df.iterrows():
            conf = (
                float(row[confidence_col])
                if confidence_col and confidence_col in df.columns
                else 1.0
            )
            self.add_sample(str(row[image_col]), str(row[label_col]), conf)

        added = len(self._records) - before
        logger.info("Loaded %d samples from %s", added, csv_path)
        return added

    def load_from_json(self, json_path: str) -> int:
        """
        Load from a JSON array of objects with keys
        ``image_path``, ``label``, and optionally ``confidence``.

        Returns the number of records added.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        before = len(self._records)
        for item in data:
            self.add_sample(
                item["image_path"],
                item["label"],
                item.get("confidence", 1.0),
            )

        added = len(self._records) - before
        logger.info("Loaded %d samples from %s", added, json_path)
        return added

    def load_from_label_file(self, label_file: str, image_dir: str) -> int:
        """
        Load from a PaddleOCR-format label file::

            <relative_image_path>\\t<label_text>

        ``image_dir`` is used to resolve relative paths.

        Returns the number of records added.
        """
        before = len(self._records)
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "\t" not in line:
                    continue
                rel_path, label = line.split("\t", 1)
                abs_path = os.path.join(image_dir, rel_path)
                self.add_sample(abs_path, label)

        added = len(self._records) - before
        logger.info("Loaded %d samples from label file %s", added, label_file)
        return added

    # ── Dataset preparation ───────────────────────────────────────────────────

    def split_train_val(
        self,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple["FinetuneDataset", "FinetuneDataset"]:
        """
        Randomly split into a training and a validation ``FinetuneDataset``.

        Both returned datasets share the same ``data_dir`` base.
        """
        if not self._records:
            raise ValueError("Cannot split an empty dataset.")

        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self._records))
        val_n = max(1, int(len(self._records) * val_ratio))

        val_set = set(indices[:val_n].tolist())

        train_ds = FinetuneDataset(str(self.data_dir / "train"))
        val_ds = FinetuneDataset(str(self.data_dir / "val"))

        for i, rec in enumerate(self._records):
            if i in val_set:
                val_ds._records.append(rec)
            else:
                train_ds._records.append(rec)

        logger.info(
            "Split: %d train / %d val (seed=%d)",
            len(train_ds),
            len(val_ds),
            seed,
        )
        return train_ds, val_ds

    def copy_images(self, target_dir: str) -> "FinetuneDataset":
        """
        Copy all images into *target_dir* and return a new ``FinetuneDataset``
        whose records point to the copied files.

        Useful for creating a self-contained training directory.
        """
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        new_ds = FinetuneDataset(target_dir)

        for rec in self._records:
            dst = target / Path(rec.image_path).name
            if not dst.exists():
                shutil.copy2(rec.image_path, dst)
            new_ds._records.append(
                AnnotationRecord(str(dst), rec.label, rec.confidence)
            )

        return new_ds

    def save_label_file(
        self,
        label_file: str,
        relative_to: Optional[str] = None,
    ) -> None:
        """
        Write a PaddleOCR-format label file.

        If *relative_to* is given, image paths in the file will be written
        relative to that directory (cross-drive paths remain absolute on Windows).
        """
        Path(label_file).parent.mkdir(parents=True, exist_ok=True)
        with open(label_file, "w", encoding="utf-8") as f:
            for rec in self._records:
                img_path = rec.image_path
                if relative_to:
                    try:
                        img_path = os.path.relpath(img_path, relative_to)
                    except ValueError:
                        pass  # relpath fails across drives on Windows
                # Always use forward slashes so PaddleOCR can read on all OSes.
                img_path = img_path.replace("\\", "/")
                f.write(f"{img_path}\t{rec.label}\n")

        logger.info("Saved %d labels to %s", len(self._records), label_file)

    def save_annotations_json(self, json_path: str) -> None:
        """
        Dump all annotations to a human-readable JSON file for inspection
        and manual correction.
        """
        data = [
            {
                "image_path": r.image_path,
                "label": r.label,
                "confidence": r.confidence,
            }
            for r in self._records
        ]
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("Saved %d annotations to %s", len(self._records), json_path)

    def build_char_dict(self, output_path: str) -> List[str]:
        """
        Build a character dictionary from all labels and write one character
        per line to *output_path*.

        Returns the sorted list of unique characters.
        """
        chars: set = set()
        for rec in self._records:
            chars.update(rec.label)

        char_list = sorted(chars)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for ch in char_list:
                f.write(ch + "\n")

        logger.info(
            "Built char dict with %d chars → %s", len(char_list), output_path
        )
        return char_list

    # ── Data cleaning ─────────────────────────────────────────────────────────

    def filter_empty_labels(self) -> int:
        """
        Remove records whose label is an empty string.

        Empty labels (blank table cells) do not contribute useful signal for
        text recognition training and can confuse the CTC decoder.  Call this
        on both the training and validation datasets before handing them to
        ``PaddleOCRVLFinetuner.run()``.

        Returns the number of records removed.
        """
        before = len(self._records)
        self._records = [r for r in self._records if r.label]
        removed = before - len(self._records)
        if removed:
            logger.info(
                "filter_empty_labels: removed %d empty-label records (%d remaining).",
                removed,
                len(self._records),
            )
        return removed

    # ── Statistics ────────────────────────────────────────────────────────────

    def statistics(self) -> dict:
        """Return a summary dict describing the dataset."""
        if not self._records:
            return {"total": 0}

        lengths = [len(r.label) for r in self._records]
        unique_chars: set = set()
        for r in self._records:
            unique_chars.update(r.label)

        return {
            "total": len(self._records),
            "unique_chars": len(unique_chars),
            "avg_label_length": round(sum(lengths) / len(lengths), 2),
            "max_label_length": max(lengths),
            "min_label_length": min(lengths),
            "avg_confidence": round(
                sum(r.confidence for r in self._records) / len(self._records), 4
            ),
        }

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return (
            f"FinetuneDataset(records={len(self._records)}, "
            f"dir={self.data_dir})"
        )
