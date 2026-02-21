"""
Training data pipeline for PaddleOCR recognition fine-tuning.

Orchestrates the full flow:
  source image (GIF/PNG/…) + Excel ground truth
    → cell detection
    → crop extraction
    → label assignment
    → FinetuneDataset

Typical usage
-------------
>>> from app.infrastructure.ocr_engines.finetune.pipeline import TrainingDataPipeline
>>> pipeline = TrainingDataPipeline(output_dir="finetune_data")
>>> pipeline.add_source(
...     image_path="sample input/HGF-1500-1887.gif",
...     ground_truth_path="client data/1.HGF-1500-1887_Client_File.xlsx",
... )
>>> train_ds, val_ds = pipeline.build()
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from .dataset import AnnotationRecord, FinetuneDataset
except ImportError:
    from dataset import AnnotationRecord, FinetuneDataset  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Column schema — positional order matches the table left-to-right layout
# ──────────────────────────────────────────────────────────────────────────────

# Maps column index (0-based) to the schema field name used in COLUMN_SCHEMA
# and in the normalised Excel DataFrame.
_COL_IDX_TO_FIELD: List[str] = [
    "slno",
    "pa_first_name", "pa_last_name", "pa_address", "pa_email",
    "pa_phone", "pa_dob", "pa_city", "pa_state", "pa_zipcode",
    "pa_medicare_claim_number",
    "dr_first_name", "dr_last_name", "dr_address", "dr_phone",
    "dr_city", "dr_email",
    "ag_first_name", "ag_last_name", "ag_phone", "ag_address",
    "ag_state", "ag_zipcode", "ag_dob", "ag_email",
]

# Maps common Excel column header variants (lowercased, stripped) → schema field
_EXCEL_TO_FIELD: dict = {
    "slno": "slno", "sl no": "slno", "serial no": "slno",
    "pa first name": "pa_first_name",
    "pa last name": "pa_last_name",
    "pa address": "pa_address",
    "pa email id": "pa_email", "pa email": "pa_email",
    "pa phone": "pa_phone",
    "pa dob": "pa_dob",
    "pa city": "pa_city",
    "pa state": "pa_state",
    "pa zipcode": "pa_zipcode", "pa zip code": "pa_zipcode",
    "pa medicare claim number": "pa_medicare_claim_number",
    "dr name": "dr_first_name", "dr first name": "dr_first_name",
    "dr last name": "dr_last_name",
    "dr address": "dr_address", "dr physical address": "dr_address",
    "dr phone": "dr_phone",
    "dr city": "dr_city",
    "dr email id": "dr_email", "dr email": "dr_email",
    "ag first name": "ag_first_name",
    "ag last name": "ag_last_name",
    "ag phone": "ag_phone",
    "ag physical address": "ag_address", "ag address": "ag_address",
    "ag state": "ag_state",
    "ag zip code": "ag_zipcode", "ag zipcode": "ag_zipcode",
    "ag dob": "ag_dob",
    "ag email id": "ag_email", "ag email": "ag_email",
}


# ──────────────────────────────────────────────────────────────────────────────
# Internal data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _Source:
    image_path: str
    ground_truth_path: str
    sheet_name: int | str
    slno_col: str


@dataclass
class _Cell:
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2) axis-aligned
    score: float


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class TrainingDataPipeline:
    """
    End-to-end pipeline: source images + Excel ground truth → FinetuneDataset.

    Steps
    -----
    1. Convert each source image (GIF / PNG / JPG / …) to PNG frame(s).
    2. Run PaddleOCR detection to locate all text-cell bounding boxes.
    3. Cluster detected boxes into table rows by y-coordinate proximity.
    4. Crop each cell (with optional padding); save to ``output_dir/crops/``.
    5. Load the Excel ground truth and normalise its column names.
    6. Match each detected table row to an Excel row — primarily by SLNO
       value, with a positional fallback.
    7. Assign each crop its ground-truth label using the column-schema order.
    8. Return populated ``FinetuneDataset`` objects (train + val split).

    Parameters
    ----------
    output_dir:
        Root directory for all pipeline outputs (crops, label files, stats).
    min_detection_score:
        PaddleOCR detection confidence threshold; boxes below this are dropped.
    row_gap_ratio:
        Controls row clustering sensitivity.  Two boxes whose y-centres differ
        by more than ``row_gap_ratio * median_box_height`` are placed in
        separate rows.  Increase for tables with tall rows; decrease for dense
        layouts.
    cell_padding:
        Extra pixels added around each cropped bounding box.
    skip_header_rows:
        Number of detected rows at the top of the image to treat as headers
        (and exclude from training samples).
    min_crop_px:
        Crops whose width *or* height is below this threshold are discarded
        (avoids saving degenerate slices).
    use_grid_slicing:
        When *True* (default), use OpenCV morphological line detection to
        locate the table grid and slice every cell directly from its grid
        position — no PaddleOCR detection needed.  Each (row, col) crop is
        mapped 1-to-1 to the corresponding Excel cell, which is reliable and
        efficient for fixed-format scanned tables.
        When *False*, fall back to the original PaddleOCR detection +
        row-clustering approach.
    expected_table_cols:
        Number of columns the grid slicer should expect.  Defaults to 25
        (SLNO + 24 data columns).  If fewer vertical lines are detected the
        pipeline falls back to a uniform column partition.
    crop_upscale:
        Scale factor applied to each saved cell crop image.  Defaults to 2.0
        (double the extracted pixel dimensions using bicubic interpolation).
        Higher values produce larger, sharper training images from dense scans
        where raw row height is only ~20 px.  Set to 1.0 to disable upscaling.
    """

    def __init__(
        self,
        output_dir: str = "finetune_data",
        min_detection_score: float = 0.0,
        row_gap_ratio: float = 0.5,
        cell_padding: int = 4,
        skip_header_rows: int = 1,
        min_crop_px: int = 8,
        use_grid_slicing: bool = True,
        expected_table_cols: int = 25,
        crop_upscale: float = 2.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.crops_dir = self.output_dir / "crops"
        self.crops_dir.mkdir(parents=True, exist_ok=True)

        self.min_detection_score = min_detection_score
        self.row_gap_ratio = row_gap_ratio
        self.cell_padding = cell_padding
        self.skip_header_rows = skip_header_rows
        self.min_crop_px = min_crop_px
        self.use_grid_slicing = use_grid_slicing
        self.expected_table_cols = expected_table_cols
        self.crop_upscale = crop_upscale

        self._sources: List[_Source] = []
        self._all_records: List[AnnotationRecord] = []
        self._ocr = None  # lazy-initialised PaddleOCR instance

    # ── Public API ────────────────────────────────────────────────────────────

    def add_source(
        self,
        image_path: str,
        ground_truth_path: str,
        sheet_name: int | str = 0,
        slno_col: str = "SLNO",
    ) -> "TrainingDataPipeline":
        """
        Register a (source image, Excel ground truth) pair.

        Can be called multiple times; all sources are processed together
        when ``build()`` is called.

        Returns *self* so calls can be chained::

            pipeline.add_source(...).add_source(...)
        """
        self._sources.append(_Source(image_path, ground_truth_path, sheet_name, slno_col))
        return self

    def build(
        self,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[FinetuneDataset, FinetuneDataset]:
        """
        Run the pipeline over all registered sources and return
        ``(train_dataset, val_dataset)``.

        Side-effects
        ------------
        Writes to ``output_dir/``:
          - ``crops/``           – individual cell PNG images
          - ``source_pngs/``     – intermediate per-frame PNGs
          - ``annotations.json`` – full annotation list (human-readable)
          - ``manifest.json``    – alias for annotations.json
          - ``train_list.txt``   – PaddleOCR label file for training
          - ``val_list.txt``     – PaddleOCR label file for validation
          - ``pipeline_stats.json`` – dataset statistics
        """
        if not self._sources:
            raise ValueError("No sources registered. Call add_source() first.")

        self._all_records.clear()

        for src in self._sources:
            records = self._process_source(src)
            self._all_records.extend(records)
            logger.info(
                "Source '%s' → %d annotation records.",
                Path(src.image_path).name,
                len(records),
            )

        if not self._all_records:
            raise RuntimeError(
                "Pipeline produced zero annotation records.  "
                "Check that the source images contain detectable text cells "
                "and that the ground-truth Excel file has matching rows."
            )

        full_ds = FinetuneDataset(str(self.output_dir))
        full_ds._records = list(self._all_records)

        train_ds, val_ds = full_ds.split_train_val(val_ratio=val_ratio, seed=seed)
        train_ds.save_label_file(str(self.output_dir / "train_list.txt"))
        val_ds.save_label_file(str(self.output_dir / "val_list.txt"))
        full_ds.save_annotations_json(str(self.output_dir / "annotations.json"))
        self._save_manifest()

        stats = full_ds.statistics()
        logger.info("Pipeline complete.  Dataset stats: %s", stats)
        (self.output_dir / "pipeline_stats.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8"
        )

        return train_ds, val_ds

    def report_ocr_errors(
        self,
        md_dir: str,
        ground_truth_path: str,
        output_csv: Optional[str] = None,
        sheet_name: int | str = 0,
    ):
        """
        Compare OCR markdown output against Excel ground truth.

        Parses all ``*.md`` files in *md_dir*, extracts cell text, and aligns
        rows with the Excel file via SLNO.  Returns a DataFrame with columns:

            slno | field | ocr_text | gt_text | match

        Parameters
        ----------
        md_dir:             Directory containing OCR output ``.md`` files.
        ground_truth_path:  Path to the Excel ground truth file.
        output_csv:         Optional path; when given, the report is saved as CSV.
        sheet_name:         Excel sheet index or name.
        """
        import pandas as pd
        from bs4 import BeautifulSoup  # type: ignore

        gt_df = self._load_ground_truth(ground_truth_path, sheet_name)
        md_files = sorted(Path(md_dir).glob("*.md"))

        rows = []
        for md_file in md_files:
            content = md_file.read_text(encoding="utf-8")
            soup = BeautifulSoup(content, "html.parser")
            table = soup.find("table")
            if not table:
                continue

            for tr_idx, tr in enumerate(table.find_all("tr")):
                if tr_idx < self.skip_header_rows:
                    continue
                cells = [td.get_text(strip=True) for td in tr.find_all("td")]
                if not cells:
                    continue

                slno = cells[0].strip() if cells else ""
                gt_row = None
                if "slno" in gt_df.columns:
                    matches = gt_df[gt_df["slno"].str.strip() == slno]
                    if not matches.empty:
                        gt_row = matches.iloc[0]

                for col_i, cell_text in enumerate(cells):
                    if col_i >= len(_COL_IDX_TO_FIELD):
                        break
                    field_name = _COL_IDX_TO_FIELD[col_i]
                    gt_text = ""
                    if gt_row is not None and field_name in gt_row.index:
                        gt_text = str(gt_row[field_name]).strip()

                    rows.append({
                        "slno": slno,
                        "field": field_name,
                        "ocr_text": cell_text.strip(),
                        "gt_text": gt_text,
                        "match": cell_text.strip() == gt_text,
                    })

        report_df = pd.DataFrame(rows)

        if output_csv:
            report_df.to_csv(output_csv, index=False)
            logger.info("OCR error report saved → %s", output_csv)

        if not report_df.empty:
            accuracy = report_df["match"].mean()
            logger.info(
                "OCR error report: %d cells across %d files, %.1f%% exact match.",
                len(report_df),
                len(md_files),
                accuracy * 100,
            )

        return report_df

    # ── Per-source orchestration ───────────────────────────────────────────────

    def _process_source(self, src: _Source) -> List[AnnotationRecord]:
        gt_df = self._load_ground_truth(src.ground_truth_path, src.sheet_name)
        png_paths = self._to_png_frames(src.image_path)

        records: List[AnnotationRecord] = []
        gt_row_offset = 0
        source_stem = Path(src.image_path).stem

        mode = "grid-slice" if self.use_grid_slicing else "ocr-detect"
        logger.info("Processing '%s' in %s mode.", Path(src.image_path).name, mode)

        for frame_idx, png_path in enumerate(png_paths):
            if self.use_grid_slicing:
                frame_records, rows_consumed = self._grid_slice_frame(
                    png_path=png_path,
                    gt_df=gt_df,
                    gt_row_offset=gt_row_offset,
                    frame_idx=frame_idx,
                    source_stem=source_stem,
                )
            else:
                frame_records, rows_consumed = self._process_frame(
                    png_path=png_path,
                    gt_df=gt_df,
                    gt_row_offset=gt_row_offset,
                    frame_idx=frame_idx,
                    source_stem=source_stem,
                )
            records.extend(frame_records)
            gt_row_offset += rows_consumed

        return records

    def _process_frame(
        self,
        png_path: str,
        gt_df,
        gt_row_offset: int,
        frame_idx: int,
        source_stem: str,
    ) -> Tuple[List[AnnotationRecord], int]:
        """Process one PNG frame: detect → cluster → crop → label."""
        cells = self._detect_cells(png_path)
        if not cells:
            logger.warning("No cells detected in '%s' (frame %d).", png_path, frame_idx)
            return [], 0

        rows = self._cluster_rows(cells)
        data_rows = rows[self.skip_header_rows:]

        image = cv2.imread(png_path)
        if image is None:
            logger.error("cv2 could not read '%s'.", png_path)
            return [], 0

        records: List[AnnotationRecord] = []
        matched_rows = 0

        for row_i, row_cells in enumerate(data_rows):
            gt_idx = gt_row_offset + row_i
            if gt_idx >= len(gt_df):
                logger.debug(
                    "Ground-truth exhausted at data row %d (frame %d).",
                    row_i, frame_idx,
                )
                break

            gt_row = gt_df.iloc[gt_idx]

            # Optional SLNO cross-check: warn if the first cell doesn't match.
            if row_cells and "slno" in gt_df.columns:
                expected_slno = str(gt_row.get("slno", "")).strip()
                # (We do not abort on mismatch — positional alignment is primary.)

            for col_i, cell in enumerate(row_cells):
                if col_i >= len(_COL_IDX_TO_FIELD):
                    break

                field_name = _COL_IDX_TO_FIELD[col_i]

                # Retrieve ground-truth label safely
                gt_label = ""
                if field_name in gt_row.index:
                    raw = gt_row[field_name]
                    gt_label = "" if str(raw).lower() in ("nan", "none", "") else str(raw).strip()

                # Always include every cell (even blank ones) so all detected
                # cells are mapped to the ground-truth grid.
                crop_name = f"{source_stem}_f{frame_idx:02d}_r{row_i:04d}_c{col_i:02d}"
                crop_path = self._crop_cell(image, cell.bbox, crop_name)
                if crop_path is None:
                    continue

                records.append(
                    AnnotationRecord(
                        image_path=crop_path,
                        label=gt_label,
                        confidence=cell.score,
                    )
                )

            matched_rows += 1

        return records, matched_rows

    # ── Grid-based cell slicing ───────────────────────────────────────────────

    def _grid_slice_frame(
        self,
        png_path: str,
        gt_df,
        gt_row_offset: int,
        frame_idx: int,
        source_stem: str,
    ) -> Tuple[List[AnnotationRecord], int]:
        """
        Process one PNG frame by slicing it into a grid of cells and mapping
        each cell directly to the corresponding Excel ground-truth value.

        This is the preferred path for fixed-format scanned tables (e.g. a
        1500-row × 25-column sheet rendered as a single GIF page).  It avoids
        the OCR detection + row-clustering step entirely:

        1. Load the PNG frame.
        2. Detect horizontal / vertical grid lines with morphological ops.
        3. Derive (y1, y2) row spans and (x1, x2) column spans.
        4. Skip the first ``skip_header_rows`` detected rows.
        5. For each remaining (row, col) cell:
             a. Look up the ground-truth label from gt_df by positional index.
             b. Crop the cell region (with ``cell_padding`` border pixels).
             c. Save the crop and create an AnnotationRecord.
        """
        image = cv2.imread(png_path)
        if image is None:
            logger.error("cv2 could not read '%s'.", png_path)
            return [], 0

        remaining_gt_rows = len(gt_df) - gt_row_offset
        if remaining_gt_rows <= 0:
            logger.debug("No ground-truth rows remaining for frame %d.", frame_idx)
            return [], 0

        expected_rows = remaining_gt_rows + self.skip_header_rows
        col_spans, row_spans = self._detect_grid_boundaries(
            image,
            expected_cols=self.expected_table_cols,
            expected_rows=expected_rows,
        )

        # Remove degenerate spans (scan artefacts such as a thin top/bottom
        # border) before applying skip_header_rows.  Without this filter the
        # tiny border span is counted as "row 0" and skip_header_rows ends up
        # skipping the border instead of the real column-header row, causing
        # every image crop to be labelled with the wrong (next-row) ground-
        # truth value and the last data row to be missing entirely.
        row_spans = [
            (y1, y2) for y1, y2 in row_spans if y2 - y1 >= self.min_crop_px
        ]

        data_row_spans = row_spans[self.skip_header_rows:]

        records: List[AnnotationRecord] = []
        matched_rows = 0

        for row_i, (ry1, ry2) in enumerate(data_row_spans):
            gt_idx = gt_row_offset + row_i
            if gt_idx >= len(gt_df):
                logger.debug(
                    "Ground-truth exhausted at data row %d (frame %d).",
                    row_i,
                    frame_idx,
                )
                break

            gt_row = gt_df.iloc[gt_idx]

            for col_i, (cx1, cx2) in enumerate(col_spans):
                if col_i >= len(_COL_IDX_TO_FIELD):
                    break

                field_name = _COL_IDX_TO_FIELD[col_i]

                gt_label = ""
                if field_name in gt_row.index:
                    raw = gt_row[field_name]
                    gt_label = (
                        ""
                        if str(raw).lower() in ("nan", "none", "")
                        else str(raw).strip()
                    )

                # Always include every cell so the full 1500 × 25 = 37 500
                # grid is mapped. Empty cells are saved with label "" so
                # the model can learn to recognise blank table cells.
                crop_name = (
                    f"{source_stem}_f{frame_idx:02d}_r{row_i:04d}_c{col_i:02d}"
                )
                crop_path = self._crop_cell(image, (cx1, ry1, cx2, ry2), crop_name)
                if crop_path is None:
                    continue

                records.append(
                    AnnotationRecord(
                        image_path=crop_path,
                        label=gt_label,
                        confidence=1.0,  # grid-derived; no OCR detection score
                    )
                )

            matched_rows += 1

        logger.info(
            "Frame %d: grid-sliced %d data rows × %d cols → %d annotation records.",
            frame_idx,
            len(data_row_spans),
            len(col_spans),
            len(records),
        )
        return records, matched_rows

    def _detect_grid_boundaries(
        self,
        image: np.ndarray,
        expected_cols: int = 25,
        expected_rows: Optional[int] = None,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Detect column and row boundaries using projection-based gap / valley
        analysis — robust for scanned tables that have no drawn grid lines.

        Algorithm
        ---------
        Row boundaries
            Threshold the image to isolate dark ink pixels.  Compute the
            fraction of dark pixels in each row.  Rows where fewer than 2 %
            of pixels are dark are "gap rows" (the whitespace between table
            rows).  Consecutive gap-row clusters are collapsed to their
            median → one boundary per inter-row gap.

        Column boundaries
            Restrict to *data rows* (all rows that are NOT gap rows) to
            eliminate noise from the whitespace bands.  Compute the fraction
            of dark pixels in each column over those data rows.  Columns
            where fewer than 4 % of data-row pixels are dark are "column
            gap" columns (the whitespace between table columns).
            Consecutive gap-column clusters are collapsed to their median →
            one boundary per inter-column gap.  Boundaries that are closer
            together than ``w / (expected_cols * 3)`` pixels are filtered
            out as noise.

        Fallbacks
            If the detected number of row or column spans is less than
            expected, a uniform partition is used for that axis and a
            warning is logged.

        Returns
        -------
        col_spans : list of (x1, x2) tuples, sorted left → right
        row_spans : list of (y1, y2) tuples, sorted top  → bottom
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Threshold: dark ink → white (1), bright background → black (0)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        binary_f = binary.astype(np.float32) / 255.0  # values in [0, 1]

        # ── Row boundary detection — find whitespace gaps ─────────────────
        row_frac = binary_f.sum(axis=1) / w   # dark fraction per row
        gap_row_idx = np.where(row_frac < 0.02)[0]

        row_bounds = self._merge_line_positions(gap_row_idx, min_gap=3)

        if len(row_bounds) >= 2:
            rb = [0] + row_bounds + [h - 1]
            row_spans: List[Tuple[int, int]] = [
                (rb[i], rb[i + 1]) for i in range(len(rb) - 1)
            ]
        else:
            n_rows = expected_rows if expected_rows else 1501
            logger.warning(
                "Row gap detection found only %d boundaries; "
                "falling back to uniform row grid (%d spans).",
                len(row_bounds), n_rows,
            )
            row_spans = self._uniform_spans(h, n_rows)

        # ── Column boundary detection — valley analysis on data rows ──────
        # Restrict to data rows (not gap rows) so whitespace bands don't
        # distort the column projection.
        data_mask = np.ones(h, dtype=bool)
        if len(gap_row_idx) > 0:
            data_mask[gap_row_idx] = False
        data_row_idx = np.where(data_mask)[0]

        if len(data_row_idx) > 0:
            col_frac = binary_f[data_row_idx, :].sum(axis=0) / len(data_row_idx)
        else:
            col_frac = binary_f.sum(axis=0) / h

        # Valley (near-zero) columns = whitespace between columns
        col_gap_idx = np.where(col_frac < 0.04)[0]
        col_bounds = self._merge_line_positions(col_gap_idx, min_gap=10)

        # Minimum distance between two valid column boundaries.
        # Narrower than this → the "span" is noise, not a real column.
        min_span_w = max(15, w // (expected_cols * 3))

        # Drop boundaries that sit within the left/right image margin
        # (often artefacts of scanning frame) and enforce minimum spacing.
        margin = max(5, min_span_w // 3)
        filtered_bounds: List[int] = []
        for b in col_bounds:
            if b <= margin or b >= w - margin:
                continue  # skip boundary at image edge
            if not filtered_bounds or b - filtered_bounds[-1] >= min_span_w:
                filtered_bounds.append(b)
        col_bounds = filtered_bounds

        if len(col_bounds) >= expected_cols - 1:
            cb = [0] + col_bounds + [w - 1]
            col_spans: List[Tuple[int, int]] = [
                (cb[i], cb[i + 1]) for i in range(len(cb) - 1)
            ]
            # Discard spans that are implausibly narrow (scan noise) and
            # trim excess by keeping only the widest expected_cols spans.
            col_spans = [s for s in col_spans if s[1] - s[0] >= min_span_w]
            if len(col_spans) > expected_cols:
                col_spans = sorted(
                    sorted(col_spans, key=lambda s: s[1] - s[0], reverse=True)[
                        :expected_cols
                    ],
                    key=lambda s: s[0],
                )
        else:
            logger.warning(
                "Column valley detection found %d boundaries (expected ≥ %d); "
                "falling back to uniform column grid.",
                len(col_bounds), expected_cols - 1,
            )
            col_spans = self._uniform_spans(w, expected_cols)

        logger.info(
            "Grid: %d col spans × %d row spans  (image %d × %d px).",
            len(col_spans), len(row_spans), w, h,
        )
        return col_spans, row_spans

    @staticmethod
    def _merge_line_positions(
        positions: np.ndarray, min_gap: int = 5
    ) -> List[int]:
        """
        Collapse clusters of adjacent pixel positions into one representative
        position per cluster (integer median of each cluster).

        Parameters
        ----------
        positions : Sorted 1-D array of pixel indices from np.where().
        min_gap   : Maximum gap (px) between positions in the same cluster.
        """
        if len(positions) == 0:
            return []
        result: List[int] = []
        cluster: List[int] = [int(positions[0])]
        for p in positions[1:]:
            if p - cluster[-1] <= min_gap:
                cluster.append(int(p))
            else:
                result.append(int(np.median(cluster)))
                cluster = [int(p)]
        result.append(int(np.median(cluster)))
        return result

    @staticmethod
    def _uniform_spans(total_size: int, n: int) -> List[Tuple[int, int]]:
        """
        Divide *total_size* pixels evenly into *n* (start, end) span tuples.

        Used as a fallback when morphological line detection does not find
        the expected number of grid lines.
        """
        step = total_size / n
        return [(int(i * step), int((i + 1) * step)) for i in range(n)]

    # ── Image conversion ──────────────────────────────────────────────────────

    def _to_png_frames(self, image_path: str) -> List[str]:
        """
        Convert a source image to one or more PNG files.

        - Animated GIFs and multi-page TIFFs → one PNG per frame.
        - All other formats                   → single PNG.

        Frames that already exist on disk are reused (idempotent).
        """
        from PIL import Image as PILImage

        src = Path(image_path)
        out_dir = self.output_dir / "source_pngs"
        out_dir.mkdir(parents=True, exist_ok=True)

        ext = src.suffix.lower()
        is_multi = ext in (".gif", ".tiff", ".tif")

        if not is_multi:
            single_out = out_dir / f"{src.stem}.png"
            if not single_out.exists():
                with PILImage.open(str(src)) as img:
                    img.convert("RGB").save(str(single_out))
                logger.info("Converted '%s' → '%s'.", src.name, single_out.name)
            return [str(single_out)]

        # Multi-frame path
        paths: List[str] = []
        with PILImage.open(str(src)) as img:
            for frame_idx in range(getattr(img, "n_frames", 1)):
                out_path = out_dir / f"{src.stem}_frame{frame_idx:03d}.png"
                if not out_path.exists():
                    img.seek(frame_idx)
                    img.convert("RGB").save(str(out_path))
                paths.append(str(out_path))

        logger.info("Extracted %d frame(s) from '%s'.", len(paths), src.name)
        return paths

    # ── Cell detection ────────────────────────────────────────────────────────

    def _get_ocr(self):
        """Lazy-initialise the PaddleOCR detection engine (single instance)."""
        if self._ocr is None:
            from paddleocr import PaddleOCR  # type: ignore
            self._ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
        return self._ocr

    def _detect_cells(self, image_path: str) -> List[_Cell]:
        """
        Run PaddleOCR in detection-only mode and return axis-aligned bounding
        boxes with their detection scores.

        Handles both the old ``ocr()`` result format (list of
        ``[quad, score]`` pairs) and the newer structured result objects.
        """
        try:
            result = self._get_ocr().ocr(image_path, det=True, rec=False, cls=False)
        except Exception as exc:
            logger.error("PaddleOCR detection failed for '%s': %s", image_path, exc)
            return []

        if not result or not result[0]:
            return []

        cells: List[_Cell] = []
        for item in result[0]:
            quad, score = self._parse_detection_item(item)
            if quad is None or score < self.min_detection_score:
                continue
            xs = [pt[0] for pt in quad]
            ys = [pt[1] for pt in quad]
            bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            cells.append(_Cell(bbox=bbox, score=score))

        return cells

    @staticmethod
    def _parse_detection_item(item) -> Tuple[Optional[list], float]:
        """
        Normalise a single PaddleOCR detection result item to
        ``(quad, score)`` where *quad* is a list of four [x, y] points.

        Handles the two most common output shapes:
          - ``[[[x,y]×4], score]``   (standard paddleocr ≥ 2.x)
          - ``[[x,y]×4]``            (detection-only, no score)
        """
        if (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[0], list)
            and not isinstance(item[1], list)
        ):
            return item[0], float(item[1])

        if (
            isinstance(item, list)
            and len(item) == 4
            and isinstance(item[0], (list, tuple))
        ):
            return item, 1.0

        return None, 0.0

    # ── Row clustering ────────────────────────────────────────────────────────

    def _cluster_rows(self, cells: List[_Cell]) -> List[List[_Cell]]:
        """
        Group cells into table rows using y-centre proximity clustering.

        Two cells are in the same row when their y-centres are within
        ``row_gap_ratio × median_cell_height``.
        """
        if not cells:
            return []

        heights = [c.bbox[3] - c.bbox[1] for c in cells]
        med_h = float(np.median(heights)) if heights else 20.0
        gap = max(1.0, self.row_gap_ratio * med_h)

        def y_centre(c: _Cell) -> float:
            return (c.bbox[1] + c.bbox[3]) / 2.0

        sorted_cells = sorted(cells, key=lambda c: (y_centre(c), c.bbox[0]))

        rows: List[List[_Cell]] = []
        current_row: List[_Cell] = [sorted_cells[0]]
        current_y = y_centre(sorted_cells[0])

        for cell in sorted_cells[1:]:
            cy = y_centre(cell)
            if abs(cy - current_y) <= gap:
                current_row.append(cell)
                current_y = float(np.mean([y_centre(c) for c in current_row]))
            else:
                rows.append(sorted(current_row, key=lambda c: c.bbox[0]))
                current_row = [cell]
                current_y = cy

        rows.append(sorted(current_row, key=lambda c: c.bbox[0]))
        return rows

    # ── Cell cropping ─────────────────────────────────────────────────────────

    def _crop_cell(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        name: str,
    ) -> Optional[str]:
        """
        Crop *bbox* from *image* with ``cell_padding`` border pixels and save
        to ``crops_dir/<name>.png``.

        Returns the saved file path, or *None* if the crop is degenerate.
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        pad = self.cell_padding

        x1c = max(0, x1 - pad)
        y1c = max(0, y1 - pad)
        x2c = min(w, x2 + pad)
        y2c = min(h, y2 + pad)

        crop_w = x2c - x1c
        crop_h = y2c - y1c
        if crop_w < self.min_crop_px or crop_h < self.min_crop_px:
            return None

        crop = image[y1c:y2c, x1c:x2c]

        # Upscale the crop for better training image quality.
        # Dense scans often have row heights of only ~20 px; upscaling
        # produces sharper, higher-resolution training images.
        if self.crop_upscale > 1.0:
            new_w = max(1, int(crop_w * self.crop_upscale))
            new_h = max(1, int(crop_h * self.crop_upscale))
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        out_path = self.crops_dir / f"{name}.png"
        cv2.imwrite(str(out_path), crop)
        return str(out_path)

    # ── Ground-truth loading ──────────────────────────────────────────────────

    def _load_ground_truth(self, xlsx_path: str, sheet_name: int | str = 0):
        """
        Read the Excel file, normalise column names to schema field names, and
        return a pandas DataFrame with ``dtype=str``.
        """
        import pandas as pd

        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, dtype=str)
        df = df.fillna("")

        # Map Excel headers to schema field names
        rename = {}
        for col in df.columns:
            key = col.strip().lower()
            if key in _EXCEL_TO_FIELD:
                rename[col] = _EXCEL_TO_FIELD[key]

        df = df.rename(columns=rename)

        # Auto-detect SLNO column if not already mapped
        if "slno" not in df.columns:
            for col in df.columns:
                if df[col].astype(str).str.match(r"^HGF\d+").any():
                    df = df.rename(columns={col: "slno"})
                    logger.info("Auto-detected SLNO column: '%s'.", col)
                    break

        logger.info(
            "Loaded ground truth from '%s': %d rows, fields: %s",
            Path(xlsx_path).name,
            len(df),
            [c for c in df.columns if c in set(_COL_IDX_TO_FIELD)],
        )
        return df

    # ── Persistence helpers ───────────────────────────────────────────────────

    def _save_manifest(self) -> None:
        """Write a human-readable JSON manifest of all annotation records."""
        manifest = [
            {
                "image_path": r.image_path,
                "label": r.label,
                "confidence": r.confidence,
            }
            for r in self._all_records
        ]
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Manifest saved → %s/manifest.json", self.output_dir)
