#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

uv run python paper/scripts/make_paper_assets.py

cd "${REPO_ROOT}/paper"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode main.tex
elif command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode main.tex
  bibtex main
  pdflatex -interaction=nonstopmode main.tex
  pdflatex -interaction=nonstopmode main.tex
else
  echo "LaTeX toolchain not available. Generated tables and figures only."
  exit 0
fi

echo "Built paper/main.pdf"
