# ══════════════════════════════════════════════════════════════════════════════
# PaddleOCR Fine-tuning — Makefile
# ══════════════════════════════════════════════════════════════════════════════
#
# Common targets:
#
#   make install              Install CPU PaddlePaddle + all requirements
#   make install-gpu          Install GPU PaddlePaddle (CUDA 11.8) + requirements
#   make download-pretrained  Download PP-OCRv4 English mobile pretrained weights
#   make build-dataset        Run data pipeline only (no training)
#   make train                Full pipeline: dataset → train → export
#   make train-only           Train using existing dataset (skip pipeline)
#   make export               Export best checkpoint to inference format
#   make eval                 Evaluate exported model on validation set
#   make clean-runs           Remove all training artifacts
#   make clean-data           Remove generated finetune_data/ directory
#   make lint                 Run flake8 on source files
#
# ══════════════════════════════════════════════════════════════════════════════

PYTHON      ?= python3
VENV        ?= .venv
RUN_DIR     ?= runs/finetune_v1
DATA_DIR    ?= finetune_data
PRETRAINED  ?= pretrained/en_PP-OCRv4_rec_train/best_accuracy

.PHONY: all install install-gpu download-pretrained build-dataset train \
        train-only export eval clean-runs clean-data lint help

# ── Default target ────────────────────────────────────────────────────────────
all: help

# ── Installation ──────────────────────────────────────────────────────────────

install:
	@echo "Installing CPU PaddlePaddle 2.6.1 …"
	$(PYTHON) -m pip install paddlepaddle==2.6.1 \
		-i https://pypi.tuna.tsinghua.edu.cn/simple
	@echo "Installing project requirements …"
	$(PYTHON) -m pip install -r requirements.txt

install-gpu:
	@echo "Installing GPU PaddlePaddle 2.6.1 (CUDA 11.8) …"
	$(PYTHON) -m pip install paddlepaddle-gpu==2.6.1.post118 \
		-f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
	@echo "Installing project requirements …"
	$(PYTHON) -m pip install -r requirements.txt

# ── Pretrained model ──────────────────────────────────────────────────────────

download-pretrained:
	@echo "Downloading PP-OCRv4 English mobile pretrained weights …"
	$(PYTHON) scripts/download_pretrained.py --model mobile

download-pretrained-server:
	@echo "Downloading PP-OCRv4 English server pretrained weights …"
	$(PYTHON) scripts/download_pretrained.py --model server

# ── Dataset pipeline ──────────────────────────────────────────────────────────

build-dataset:
	@echo "Running data pipeline (dataset build only) …"
	$(PYTHON) train.py --no-export --no-eval --output-dir $(RUN_DIR)

# ── Training ──────────────────────────────────────────────────────────────────

train:
	@echo "Running full fine-tuning pipeline (dataset → train → export) …"
	$(PYTHON) train.py \
		--pretrained-model $(PRETRAINED) \
		--output-dir $(RUN_DIR) \
		--epochs 50 \
		--lr 5e-5 \
		--warmup-epochs 5 \
		--max-text-length 50

train-only:
	@echo "Training with existing dataset (skipping pipeline build) …"
	$(PYTHON) -m tools.train -c $(RUN_DIR)/rec_finetune.yml

train-resume:
	@echo "Resuming training from latest checkpoint …"
	$(PYTHON) train.py \
		--pretrained-model $(RUN_DIR)/models/latest \
		--output-dir $(RUN_DIR) \
		--epochs 50 \
		--lr 5e-5

# ── Export ────────────────────────────────────────────────────────────────────

export:
	@echo "Exporting best checkpoint to inference format …"
	$(PYTHON) -c "\
import sys; sys.path.insert(0, '.'); \
from config import FinetuneConfig; \
from trainer import PaddleOCRVLFinetuner; \
t = PaddleOCRVLFinetuner(FinetuneConfig(output_dir='$(RUN_DIR)')); \
t.export()"

# ── Evaluation ────────────────────────────────────────────────────────────────

eval:
	@echo "Evaluating exported model on validation set …"
	$(PYTHON) tools/eval.py -c $(RUN_DIR)/rec_finetune.yml \
		-o Global.pretrained_model=$(RUN_DIR)/models/best_accuracy \
		   Global.checkpoints=

# ── Inference ─────────────────────────────────────────────────────────────────

infer:
	@echo "Running inference (set IMG= to specify an image):"
	@echo "  make infer IMG=path/to/image.png"
	$(PYTHON) tools/infer_rec.py \
		-c $(RUN_DIR)/rec_finetune.yml \
		-o Global.pretrained_model=$(RUN_DIR)/exported \
		   Global.infer_img=$(IMG)

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean-runs:
	@echo "Removing training artifacts in $(RUN_DIR) …"
	rm -rf $(RUN_DIR)/models $(RUN_DIR)/results $(RUN_DIR)/exported
	@echo "Keeping YAML config and dataset stats."

clean-data:
	@echo "Removing generated dataset in $(DATA_DIR) …"
	rm -rf $(DATA_DIR)

# ── Code quality ──────────────────────────────────────────────────────────────

lint:
	$(PYTHON) -m flake8 config.py dataset.py pipeline.py trainer.py train.py \
		--max-line-length 100 \
		--ignore E203,W503

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "PaddleOCR Fine-tuning — Available targets:"
	@echo ""
	@echo "  make install              Install CPU PaddlePaddle + requirements"
	@echo "  make install-gpu          Install GPU PaddlePaddle (CUDA 11.8) + requirements"
	@echo "  make download-pretrained  Download PP-OCRv4 English pretrained weights"
	@echo "  make build-dataset        Run data pipeline only (no training)"
	@echo "  make train                Full pipeline: dataset → train → export"
	@echo "  make train-only           Train with existing dataset (skip pipeline)"
	@echo "  make train-resume         Resume training from latest checkpoint"
	@echo "  make export               Export best checkpoint to inference format"
	@echo "  make eval                 Evaluate on validation set"
	@echo "  make clean-runs           Remove training artifacts (keep dataset)"
	@echo "  make clean-data           Remove generated finetune_data/"
	@echo "  make lint                 Run flake8"
	@echo ""
	@echo "Quick start:"
	@echo "  make install"
	@echo "  make download-pretrained"
	@echo "  make train"
	@echo ""
