from dataclasses import dataclass, field
from typing import List, Optional


def _gpu_available() -> bool:
    """Return True only when a physical CUDA GPU is present and accessible."""
    try:
        import paddle  # type: ignore
        return paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    except Exception:
        return False


@dataclass
class FinetuneConfig:
    """
    Configuration for fine-tuning the PaddleOCR recognition model.

    Maps directly to PaddleOCR's YAML training config format so the trainer
    can generate a valid config without any extra translation step.

    Accuracy notes
    --------------
    * ``character_type="en_sensitive"`` preserves case — critical for names,
      IDs (e.g. "HGF34401", "Jos", "EDGE").
    * ``max_text_length=50`` covers the observed maximum label length of 45
      characters with a safety margin.
    * ``learning_rate=5e-5`` is conservative for fine-tuning from a PaddleOCR
      pretrained checkpoint; use 1e-4 only when training from scratch.
    * ``grad_clip_norm=5.0`` prevents gradient explosions on small batches.
    * ``save_epoch_step=1`` ensures the best validation checkpoint is never
      skipped on short training runs.
    """

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = "finetune_output"

    # ── Pre-trained model ─────────────────────────────────────────────────────
    # Path to a PaddleOCR model directory containing `best_accuracy.pdparams`
    # (or `model.pdparams`).  Leave None to start from the PaddleOCR default.
    pretrained_model_dir: Optional[str] = None

    # ── Training hyper-parameters ─────────────────────────────────────────────
    epochs: int = 50
    batch_size: int = 8
    # Use 5e-5 when fine-tuning from a pretrained checkpoint to avoid
    # over-writing low-level features; raise to 1e-4 for scratch training.
    learning_rate: float = 5e-5
    warmup_epoch: int = 5
    weight_decay: float = 1e-5
    # ClipGradByGlobalNorm threshold — prevents loss spikes on small batches.
    grad_clip_norm: float = 5.0

    # ── Image & model settings ────────────────────────────────────────────────
    # Height × width fed into the recognition network.
    img_height: int = 32
    img_width: int = 320
    # Maximum number of characters the model should predict per crop.
    # Set to at least max_label_length observed in your dataset + safety margin.
    # Pipeline stats show max_label_length=45 → use 50.
    max_text_length: int = 50
    # "en_sensitive" preserves case (uppercase/lowercase distinctions).
    # Switch to "ch" only if Chinese characters are present.
    character_type: str = "en_sensitive"
    # Path to a custom character-dict file.  None → built from training data.
    char_dict_path: Optional[str] = None
    # Include the space character in the output vocabulary.
    use_space_char: bool = True

    # ── Infrastructure ────────────────────────────────────────────────────────
    use_gpu: bool = field(default_factory=_gpu_available)
    num_workers: int = 4
    seed: int = 42

    # ── Logging / evaluation ──────────────────────────────────────────────────
    print_batch_step: int = 20
    # Save a checkpoint every epoch so the best validation epoch is never lost.
    save_epoch_step: int = 1
    # [start_iter, eval_every_n_iters].  Evaluate every 200 iterations once
    # training is stable (after iter 0).
    eval_batch_step: List[int] = field(default_factory=lambda: [0, 200])
    log_smooth_window: int = 20
