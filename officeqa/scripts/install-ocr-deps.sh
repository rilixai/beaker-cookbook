#!/usr/bin/env bash
# OCR / PDF toolchain for --corpus pdfs (OfficeQA Pro report, Appendix E.3).
# Without it, PDF-mode runs take hours per question at 2-3x the cost.
#
#   bash scripts/install-ocr-deps.sh            # system packages + core Python packages
#   bash scripts/install-ocr-deps.sh --full     # also the heavy OCR stacks (paddle, surya, doctr; several GB)
#   bash scripts/install-ocr-deps.sh --system-only
set -euo pipefail

FULL=0
SYSTEM_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --full) FULL=1 ;;
    --system-only) SYSTEM_ONLY=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

if command -v apt-get >/dev/null 2>&1; then
  $SUDO apt-get update
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng libtesseract-dev libleptonica-dev \
    poppler-utils ghostscript libgl1 libglib2.0-0 ripgrep imagemagick \
    ocrmypdf qpdf pdfgrep
elif command -v brew >/dev/null 2>&1; then
  brew install tesseract leptonica poppler ghostscript ripgrep imagemagick ocrmypdf qpdf pdfgrep
else
  echo "No apt-get or brew found. Install manually: tesseract-ocr tesseract-ocr-eng libtesseract-dev" >&2
  echo "libleptonica-dev poppler-utils ghostscript libgl1 libglib2.0-0 ripgrep imagemagick ocrmypdf qpdf pdfgrep" >&2
  exit 1
fi

if [ "$SYSTEM_ONLY" -eq 1 ]; then
  exit 0
fi

# Python packages go into the recipe's environment; per-question venvs are
# created with --system-site-packages so the agent sees them.
cd "$(dirname "$0")/.."
PIP=(uv pip install --python .venv/bin/python)
if [ ! -x .venv/bin/python ]; then
  uv sync --group dev
fi

"${PIP[@]}" \
  pytesseract pymupdf opencv-python-headless Pillow pdf2image pdfplumber pypdf PyPDF2 \
  "pdfminer.six" rapidocr-onnxruntime ocrmypdf openpyxl "camelot-py[base]" easyocr

# tesserocr needs the tesseract headers installed above; tolerate failure.
"${PIP[@]}" tesserocr || echo "tesserocr failed to build; pytesseract still works" >&2

if [ "$FULL" -eq 1 ]; then
  "${PIP[@]}" paddlepaddle paddleocr surya-ocr "python-doctr[torch]"
fi

echo "OCR toolchain installed. Verify: tesseract --version; pdftotext -v; rg --version"
