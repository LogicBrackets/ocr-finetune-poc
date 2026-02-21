"""
Fine-tuning layer for PaddleOCRVL.

Public API
----------
FinetuneConfig        – training hyperparameters and path settings
FinetuneDataset       – annotation collection, label-file generation
AnnotationRecord      – single (image_path, label, confidence) sample
PaddleOCRVLFinetuner  – end-to-end trainer: prepare → config → train → export
TrainingDataPipeline  – source images + Excel GT → FinetuneDataset

Typical usage
-------------
>>> from app.infrastructure.ocr_engines.finetune import (
...     FinetuneConfig, FinetuneDataset, PaddleOCRVLFinetuner,
...     TrainingDataPipeline,
... )

# 1. Build a dataset from source images + ground-truth Excel files
>>> pipeline = TrainingDataPipeline(output_dir="finetune_data")
>>> pipeline.add_source(
...     image_path="sample input/HGF-1500-1887.gif",
...     ground_truth_path="client data/1.HGF-1500-1887_Client_File.xlsx",
... )
>>> train_ds, val_ds = pipeline.build()

# 2. Fine-tune the recognition model
>>> cfg = FinetuneConfig(output_dir="runs/v1", epochs=20, batch_size=8)
>>> tuner = PaddleOCRVLFinetuner(cfg)
>>> best_model = tuner.run(train_ds, val_ds)   # trains and returns best checkpoint
>>> export_dir  = tuner.export(best_model)      # converts to inference format
"""

from .config import FinetuneConfig
from .dataset import AnnotationRecord, FinetuneDataset
from .pipeline import TrainingDataPipeline
from .trainer import PaddleOCRVLFinetuner

__all__ = [
    "FinetuneConfig",
    "FinetuneDataset",
    "AnnotationRecord",
    "PaddleOCRVLFinetuner",
    "TrainingDataPipeline",
]
