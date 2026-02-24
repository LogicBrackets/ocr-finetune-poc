#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# setup.sh — One-shot environment setup for PaddleOCR fine-tuning
# ══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   bash setup.sh          # CPU mode (default)
#   bash setup.sh cpu      # CPU mode
#   bash setup.sh cu117    # NVIDIA GPU, CUDA 11.7
#   bash setup.sh cu118    # NVIDIA GPU, CUDA 11.8
#
# What this script does:
#   1. Creates a Python virtual environment (.venv)
#   2. Installs PaddlePaddle (CPU or GPU variant)
#   3. Installs project requirements
#   4. Optionally downloads PP-OCRv4 pretrained weights
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
MODE="${1:-cpu}"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # No colour

info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warning() { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── Python version check ──────────────────────────────────────────────────────
PYTHON_CMD=""
for py in python3.10 python3.9 python3; do
    if command -v "$py" &>/dev/null; then
        PYTHON_CMD="$py"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    error "Python 3.9 or 3.10 is required but not found on PATH."
fi

PY_VERSION=$("$PYTHON_CMD" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Using Python $PY_VERSION  ($PYTHON_CMD)"

if [[ "$PY_VERSION" != "3.9" && "$PY_VERSION" != "3.10" ]]; then
    warning "PaddlePaddle officially supports Python 3.9 and 3.10.  " \
            "Detected $PY_VERSION — proceed with caution."
fi

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment in $VENV_DIR …"
    "$PYTHON_CMD" -m venv "$VENV_DIR"
else
    info "Virtual environment already exists: $VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python"

"$PIP" install --upgrade pip setuptools wheel -q

# ── PaddlePaddle installation ─────────────────────────────────────────────────
case "$MODE" in
    cpu)
        info "Installing PaddlePaddle 2.6.1 (CPU) …"
        "$PIP" install paddlepaddle==2.6.1 \
            -i https://pypi.tuna.tsinghua.edu.cn/simple
        ;;
    cu117)
        info "Installing PaddlePaddle 2.6.1 (GPU, CUDA 11.7) …"
        "$PIP" install paddlepaddle-gpu==2.6.1.post117 \
            -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
        ;;
    cu118)
        info "Installing PaddlePaddle 2.6.1 (GPU, CUDA 11.8) …"
        "$PIP" install paddlepaddle-gpu==2.6.1.post118 \
            -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
        ;;
    *)
        error "Unknown mode '$MODE'.  Use: cpu | cu117 | cu118"
        ;;
esac

# Verify PaddlePaddle installed correctly
"$PYTHON" -c "import paddle; print(f'PaddlePaddle {paddle.__version__} installed OK')" \
    || error "PaddlePaddle import failed.  Check the installation output above."

# ── Project requirements ───────────────────────────────────────────────────────
info "Installing project requirements …"
"$PIP" install -r "$SCRIPT_DIR/requirements.txt"

# Verify PaddleOCR
"$PYTHON" -c "import paddleocr; print(f'PaddleOCR {paddleocr.__version__} installed OK')" \
    || error "PaddleOCR import failed.  Check the installation output above."

# ── Optional: download pretrained model ───────────────────────────────────────
echo ""
read -r -p "Download PP-OCRv4 English pretrained weights now? [y/N] " DOWNLOAD
if [[ "$DOWNLOAD" =~ ^[Yy]$ ]]; then
    info "Downloading PP-OCRv4 pretrained weights …"
    "$PYTHON" "$SCRIPT_DIR/scripts/download_pretrained.py" --model mobile
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
info "Setup complete!"
echo ""
echo "  Activate the environment:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "  Start training:"
echo "    python train.py --pretrained-model pretrained/en_PP-OCRv4_rec_train/best_accuracy"
echo ""
echo "  Or use make:"
echo "    make train"
echo ""
