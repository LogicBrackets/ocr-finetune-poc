import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PaddleOCRVLFinetuner:
    """
    Fine-tunes PaddleOCR's recognition model on domain-specific document data.

    The trainer follows four steps:

    1. ``prepare_data``  – copies images into a self-contained directory and
       writes PaddleOCR-format label files plus a character dictionary.
    2. ``generate_config`` – renders a YAML training config based on
       ``FinetuneConfig`` and the prepared data paths.
    3. ``run`` – orchestrates steps 1-2 then launches PaddleOCR training
       (via its ``tools/train.py`` script) and returns the best model path.
    4. ``export`` – converts the best checkpoint into the PaddlePaddle
       inference format (``model.pdmodel`` + ``model.pdiparams``).

    Usage
    -----
    >>> from app.infrastructure.ocr_engines.finetune import (
    ...     FinetuneConfig, FinetuneDataset, PaddleOCRVLFinetuner
    ... )
    >>> cfg = FinetuneConfig(output_dir="runs/finetune_v1", epochs=20)
    >>> ds = FinetuneDataset("data/annotations")
    >>> ds.load_from_csv("train_labels.csv")
    >>> tuner = PaddleOCRVLFinetuner(cfg)
    >>> best_model = tuner.run(ds)
    >>> export_dir = tuner.export(best_model)
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self, config) -> None:
        """
        Parameters
        ----------
        config: FinetuneConfig
            Training configuration (see ``finetune.config``).
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._check_gpu()

    def _check_gpu(self) -> None:
        """Auto-detect GPU availability and fall back to CPU when unavailable."""
        if not self.config.use_gpu:
            return
        try:
            import paddle  # type: ignore

            if not paddle.is_compiled_with_cuda():
                logger.warning(
                    "GPU requested but PaddlePaddle was not compiled with CUDA "
                    "– falling back to CPU."
                )
                self.config.use_gpu = False
            elif paddle.device.cuda.device_count() == 0:
                logger.warning(
                    "PaddlePaddle is CUDA-enabled but no GPU was found "
                    "– falling back to CPU."
                )
                self.config.use_gpu = False
        except ImportError:
            logger.warning(
                "PaddlePaddle not importable – forcing CPU mode."
            )
            self.config.use_gpu = False

    # ── Step 1 – Data preparation ─────────────────────────────────────────────

    def prepare_data(
        self,
        dataset,
        val_dataset=None,
    ) -> Dict[str, Path]:
        """
        Copy images into the output directory and write label files.

        Parameters
        ----------
        dataset:     ``FinetuneDataset`` with training samples.
        val_dataset: Optional separate validation ``FinetuneDataset``.
                     When None, 15 % of *dataset* is used for validation.

        Returns a dict with keys:
          ``train_label``, ``val_label``, ``train_dir``, ``val_dir``,
          ``char_dict``.
        """
        try:
            from .dataset import FinetuneDataset  # package import
        except ImportError:
            from dataset import FinetuneDataset  # type: ignore[no-redef]  # standalone

        data_root = self.output_dir / "data"

        train_img_dir = data_root / "train_images"
        val_img_dir = data_root / "val_images"

        logger.info("Copying training images (%d samples)…", len(dataset))
        train_local = dataset.copy_images(str(train_img_dir))

        if val_dataset is None:
            logger.info(
                "No validation set provided – splitting 15 %% from training."
            )
            train_local, val_local = train_local.split_train_val(
                val_ratio=0.15, seed=self.config.seed
            )
            val_img_dir_used = train_img_dir
        else:
            logger.info("Copying validation images (%d samples)…", len(val_dataset))
            val_local = val_dataset.copy_images(str(val_img_dir))
            val_img_dir_used = val_img_dir

        train_label = data_root / "train_list.txt"
        val_label = data_root / "val_list.txt"

        # Write paths RELATIVE to each data_dir so that PaddleOCR's SimpleDataSet
        # (which joins data_dir + path-from-label-file) resolves them correctly.
        # This avoids the double-nesting bug that occurs when data_dir is absolute
        # and the label file also contains a long relative path.
        train_local.save_label_file(
            str(train_label), relative_to=str(train_img_dir)
        )
        val_local.save_label_file(
            str(val_label), relative_to=str(val_img_dir_used)
        )

        # Build character dictionary from combined samples.
        char_dict_path = data_root / "char_dict.txt"
        combined = FinetuneDataset(str(data_root))
        combined._records = list(train_local._records) + list(val_local._records)

        if self.config.char_dict_path:
            shutil.copy2(self.config.char_dict_path, char_dict_path)
            logger.info(
                "Using provided char dict: %s", self.config.char_dict_path
            )
        else:
            combined.build_char_dict(str(char_dict_path))

        stats = combined.statistics()
        (self.output_dir / "dataset_stats.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8"
        )
        logger.info("Dataset statistics: %s", stats)

        return {
            "train_label": train_label,
            "val_label": val_label,
            "train_dir": train_img_dir,
            "val_dir": val_img_dir_used,
            "char_dict": char_dict_path,
        }

    # ── Step 2 – Config generation ────────────────────────────────────────────

    def generate_config(self, data_paths: Dict[str, Path]) -> Path:
        """
        Render a PaddleOCR training YAML config and write it to
        ``<output_dir>/rec_finetune.yml``.

        The config targets the lightweight PP-OCRv4 recognition pipeline
        (MobileNetV1Enhance + SVTR neck + CTCHead), which is what
        PaddleOCRVL's recognition backbone is based on.

        All paths in the YAML are written as absolute paths so the config
        is independent of the working directory when tools/train.py runs.
        """
        cfg = self.config

        # Resolve everything to absolute paths — the subprocess runs from the
        # project root but absolute paths make the YAML self-contained.
        pretrained = str(Path(cfg.pretrained_model_dir).resolve()).replace("\\", "/") \
            if cfg.pretrained_model_dir else ""
        char_dict = str(data_paths["char_dict"].resolve()).replace("\\", "/")
        train_label = str(data_paths["train_label"].resolve()).replace("\\", "/")
        val_label = str(data_paths["val_label"].resolve()).replace("\\", "/")
        train_dir = str(data_paths["train_dir"].resolve()).replace("\\", "/")
        val_dir = str(data_paths["val_dir"].resolve()).replace("\\", "/")
        save_model_dir = str((self.output_dir / "models").resolve()).replace("\\", "/")
        save_res_path = str((self.output_dir / "results").resolve()).replace("\\", "/")
        eval_steps = str(cfg.eval_batch_step)

        use_space = str(cfg.use_space_char).lower()
        image_shape = f"[3, {cfg.img_height}, {cfg.img_width}]"

        yaml_content = f"""
Global:
  use_gpu: {str(cfg.use_gpu).lower()}
  epoch_num: {cfg.epochs}
  log_smooth_window: {cfg.log_smooth_window}
  print_batch_step: {cfg.print_batch_step}
  save_model_dir: {save_model_dir}
  save_epoch_step: {cfg.save_epoch_step}
  eval_batch_step: {eval_steps}
  cal_metric_during_train: True
  pretrained_model: {pretrained}
  checkpoints:
  use_visualdl: False
  infer_img:
  character_dict_path: {char_dict}
  character_type: {cfg.character_type}
  max_text_length: {cfg.max_text_length}
  infer_mode: False
  use_space_char: {use_space}
  save_res_path: {save_res_path}
  seed: {cfg.seed}
  distributed: False

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: {cfg.learning_rate}
    warmup_epoch: {cfg.warmup_epoch}
  regularizer:
    name: L2
    factor: {cfg.weight_decay}
  grad_clip:
    name: ClipGradByGlobalNorm
    clip_norm: {cfg.grad_clip_norm}

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform:
  Backbone:
    name: MobileNetV1Enhance
    scale: 0.5
    last_conv_stride: [1, 2]
    last_pool_type: avg
  Neck:
    name: SequenceEncoder
    encoder_type: svtr
    dims: 64
    depth: 2
    hidden_dims: 120
    use_guide: True
  Head:
    name: CTCHead
    fc_decay: {cfg.weight_decay}

Loss:
  name: CTCLoss

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc

Train:
  dataset:
    name: SimpleDataSet
    data_dir: {train_dir}
    label_file_list:
      - {train_label}
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: False
      - RecAug:
      - RecConAug:
          prob: 0.5
          image_shape: {image_shape}
      - CTCLabelEncode:
      - RecResizeImg:
          image_shape: {image_shape}
      - KeepKeys:
          keep_keys: ['image', 'label', 'length']
  loader:
    shuffle: True
    batch_size_per_card: {cfg.batch_size}
    drop_last: True
    num_workers: {cfg.num_workers}

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: {val_dir}
    label_file_list:
      - {val_label}
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: False
      - CTCLabelEncode:
      - RecResizeImg:
          image_shape: {image_shape}
      - KeepKeys:
          keep_keys: ['image', 'label', 'length']
  loader:
    shuffle: False
    drop_last: False
    batch_size_per_card: {cfg.batch_size}
    num_workers: {cfg.num_workers}
""".lstrip()

        config_path = self.output_dir / "rec_finetune.yml"
        config_path.write_text(yaml_content, encoding="utf-8")
        logger.info("Training config written: %s", config_path)
        return config_path

    # ── Step 3 – Training ─────────────────────────────────────────────────────

    def run(self, dataset, val_dataset=None) -> Path:
        """
        Full fine-tuning pipeline: prepare data → generate config → train.

        Parameters
        ----------
        dataset:     ``FinetuneDataset`` with training samples.
        val_dataset: Optional separate validation ``FinetuneDataset``.

        Returns the path to the best model directory.
        """
        logger.info("=== PaddleOCRVL Fine-tuning – START ===")

        data_paths = self.prepare_data(dataset, val_dataset)
        config_path = self.generate_config(data_paths)
        best_model_path = self._launch_training(config_path)

        logger.info(
            "=== PaddleOCRVL Fine-tuning – DONE  (best model: %s) ===",
            best_model_path,
        )
        return best_model_path

    def _launch_training(self, config_path: Path) -> Path:
        """
        Locate PaddleOCR's ``tools/train.py`` and execute it as a subprocess.

        Raises ``RuntimeError`` when the training script cannot be found
        (PaddleOCR 3.x pip package is inference-only; install the plugin with
        ``paddlex --install PaddleOCR`` to get the training scripts).
        """
        train_script = self._find_train_script()

        if train_script is None:
            raise RuntimeError(
                "PaddleOCR tools/train.py not found.\n"
                "PaddleOCR 3.x no longer ships training scripts in the pip package.\n"
                "Fix: run  paddlex --install PaddleOCR  then retry."
            )

        cmd = [sys.executable, str(train_script), "-c", str(config_path)]

        # Ensure ppocr.* and tools.* are importable in the subprocess.
        # The project root (where this file lives) must be on sys.path so that
        # both "import tools.program" and "from ppocr.data import ..." resolve.
        # tools/train.py does its own sys.path.insert but PYTHONPATH is more
        # reliable when spawning a subprocess (especially on Windows).
        project_root = str(Path(__file__).parent.resolve())
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            project_root + os.pathsep + existing if existing else project_root
        )

        logger.info("Training command: %s", " ".join(cmd))
        logger.info("use_gpu: %s", self.config.use_gpu)

        # Run from project root so that "import tools.program" and relative
        # paths in the YAML (if any remain) resolve correctly.
        result = subprocess.run(cmd, cwd=project_root, env=env)
        if result.returncode != 0:
            raise RuntimeError(
                f"PaddleOCR training failed (return code {result.returncode})."
            )

        return self._resolve_best_model()

    def _find_train_script(self) -> Optional[Path]:
        """Look for PaddleOCR's training script in common install locations.

        PaddleOCR 3.x (pip package) is inference-only and no longer ships
        ``tools/train.py``.  Training requires the PaddleOCR source repo to
        be installed via the PaddleX plugin system::

            paddlex --install PaddleOCR

        This method searches the following locations in order:

        0. Local ``tools/train.py`` bundled with this project (preferred).
        1. ``PADDLE_PDX_PADDLEOCR_PATH`` environment variable (user-set path).
        2. PaddleX-managed repo directory
           (``paddlex.repo_manager.core._GlobalContext.REPO_PARENT_DIR``).
        3. The ``paddleocr`` package root and its parent (PaddleOCR 2.x
           editable / source installs).
        """
        candidates: list[Path] = []

        # 0. Local tools/ bundled in this repository (highest priority)
        local_script = Path(__file__).parent / "tools" / "train.py"
        candidates.append(local_script)

        # 1. User-set environment variable
        env_path = os.environ.get("PADDLE_PDX_PADDLEOCR_PATH")
        if env_path:
            candidates.append(Path(env_path) / "tools" / "train.py")

        # 2. PaddleX plugin repo directory (populated by `paddlex --install PaddleOCR`)
        try:
            from paddlex.repo_manager.core import _GlobalContext  # type: ignore

            repo_parent = Path(_GlobalContext.REPO_PARENT_DIR)
            candidates.append(repo_parent / "PaddleOCR" / "tools" / "train.py")
        except Exception:
            pass

        # 3. paddleocr package root (works for PaddleOCR 2.x source installs)
        try:
            import paddleocr  # type: ignore

            paddle_root = Path(paddleocr.__file__).parent
            candidates.append(paddle_root / "tools" / "train.py")
            candidates.append(paddle_root.parent / "tools" / "train.py")
        except ImportError:
            pass

        for p in candidates:
            if p.exists():
                logger.info("Found PaddleOCR train script: %s", p)
                return p

        logger.error(
            "PaddleOCR tools/train.py not found. "
            "PaddleOCR 3.x no longer ships training scripts in the pip package. "
            "Install the training plugin with:  paddlex --install PaddleOCR"
        )
        return None

    def _resolve_best_model(self) -> Path:
        """Return the base path of the best or most recent saved checkpoint.

        PaddleOCR saves checkpoints as ``<name>.pdparams`` / ``<name>.pdopt``
        files — NOT as directories.  The base path (without extension) is what
        the export script and ``-o Global.pretrained_model=`` flag expect.

        Search order:
          1. ``models/best_accuracy.pdparams``  → return ``models/best_accuracy``
          2. most recent ``models/iter_epoch_*.pdparams``
          3. ``models/latest.pdparams`` (fallback)
          4. ``models/`` directory itself (last resort, with a warning)
        """
        models_dir = self.output_dir / "models"

        # 1. Best-accuracy checkpoint (PaddleOCR standard name)
        if (models_dir / "best_accuracy.pdparams").exists():
            return models_dir / "best_accuracy"

        # 2. Most recent epoch checkpoint
        epoch_files = sorted(
            models_dir.glob("iter_epoch_*.pdparams"),
            key=lambda p: p.stat().st_mtime,
        )
        if epoch_files:
            logger.info(
                "best_accuracy.pdparams not found; using latest epoch: %s",
                epoch_files[-1].name,
            )
            return epoch_files[-1].with_suffix("")  # strip .pdparams

        # 3. latest.pdparams (saved at end of each epoch by PaddleOCR)
        if (models_dir / "latest.pdparams").exists():
            return models_dir / "latest"

        # 4. Legacy: directory-style checkpoint (pre-2.7 format)
        best_dir = models_dir / "best_accuracy"
        if best_dir.is_dir():
            return best_dir

        logger.warning(
            "Could not locate a saved checkpoint in %s.  "
            "Has training completed successfully?",
            models_dir,
        )
        return models_dir

    # ── Step 4 – Export ───────────────────────────────────────────────────────

    def export(
        self,
        model_dir: Optional[Path] = None,
        output_dir: Optional[str] = None,
    ) -> Path:
        """
        Export a trained checkpoint to PaddlePaddle inference format
        (``model.pdmodel`` + ``model.pdiparams``).

        Parameters
        ----------
        model_dir:  checkpoint directory (defaults to ``best_accuracy``).
        output_dir: where to write the inference model (defaults to
                    ``<output_dir>/exported``).

        Returns the export directory path.
        """
        if model_dir is None:
            model_dir = self._resolve_best_model()

        export_path = Path(output_dir or str(self.output_dir / "exported"))
        export_path.mkdir(parents=True, exist_ok=True)

        export_script = self._find_export_script()
        config_path = self.output_dir / "rec_finetune.yml"

        if export_script and config_path.exists():
            cmd = [
                sys.executable,
                str(export_script),
                "-c", str(config_path),
                "-o",
                f"Global.pretrained_model={model_dir}",
                f"Global.save_inference_dir={export_path}",
            ]
            project_root = str(Path(__file__).parent.resolve())
            result = subprocess.run(cmd, cwd=project_root)
            if result.returncode != 0:
                logger.warning(
                    "Export script returned non-zero – copying weights directly."
                )
                self._copy_weights(model_dir, export_path)
        else:
            logger.info(
                "export_model.py not found – copying checkpoint weights directly."
            )
            self._copy_weights(model_dir, export_path)

        logger.info("Model exported → %s", export_path)
        return export_path

    def _find_export_script(self) -> Optional[Path]:
        # Local tools/export_model.py bundled with this project (first choice)
        local_script = Path(__file__).parent / "tools" / "export_model.py"
        if local_script.exists():
            return local_script

        # PaddleOCR 2.x pip package tools directory
        try:
            import paddleocr  # type: ignore

            script = Path(paddleocr.__file__).parent / "tools" / "export_model.py"
            if script.exists():
                return script
        except ImportError:
            pass

        return None

    def _copy_weights(self, src: Path, dst: Path) -> None:
        for pattern in ("*.pdparams", "*.pdopt", "*.pdmodel", "*.pdiparams"):
            for f in src.glob(pattern):
                shutil.copy2(f, dst / f.name)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        dataset,
        model_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Run the fine-tuned recognition model over *dataset* and return metrics.

        Parameters
        ----------
        dataset:   ``FinetuneDataset`` to evaluate.
        model_dir: path to inference-format model dir.  Defaults to
                   ``<output_dir>/exported``.

        Returns a dict with ``accuracy``, ``norm_edit_distance``, and
        ``total_samples``.
        """
        from rapidfuzz.distance import Levenshtein  # type: ignore

        resolved_model_dir = model_dir or str(self.output_dir / "exported")

        try:
            from paddleocr import PaddleOCR  # type: ignore

            ocr = PaddleOCR(
                use_angle_cls=False,
                lang="en",
                rec_model_dir=resolved_model_dir,
                show_log=False,
            )
        except Exception as exc:
            logger.error("Failed to load PaddleOCR for evaluation: %s", exc)
            return {}

        correct = 0
        edit_distances: List[float] = []

        for rec in dataset._records:
            try:
                result = ocr.ocr(rec.image_path, det=False, cls=False)
                pred = result[0][0][0] if result and result[0] else ""
            except Exception:
                pred = ""

            if pred.strip().lower() == rec.label.strip().lower():
                correct += 1

            dist = Levenshtein.distance(pred, rec.label)
            edit_distances.append(dist / max(len(rec.label), 1))

        total = len(dataset)
        metrics = {
            "accuracy": round(correct / max(total, 1), 4),
            "norm_edit_distance": round(
                1.0 - sum(edit_distances) / max(len(edit_distances), 1), 4
            ),
            "total_samples": total,
        }

        logger.info("Evaluation results: %s", metrics)
        return metrics

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def list_checkpoints(self) -> List[Path]:
        """Return all saved checkpoint directories, sorted by creation time."""
        models_dir = self.output_dir / "models"
        if not models_dir.exists():
            return []
        return sorted(
            [p for p in models_dir.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
        )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Point the config's ``pretrained_model_dir`` at an existing checkpoint
        so that the next ``run()`` call will resume from it.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )
        self.config.pretrained_model_dir = checkpoint_path
        logger.info("Checkpoint set: %s", checkpoint_path)
