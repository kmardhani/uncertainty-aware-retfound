# Paper Package

This directory contains an arXiv-compatible single-column LaTeX paper package for:

`Uncertainty-Aware Last-Layer Adaptation of RETFound for Referable Diabetic Retinopathy Screening Under Dataset Shift`

## Contents

- `main.tex`: manuscript source
- `references.bib`: bibliography
- `scripts/make_paper_assets.py`: deterministic asset generator
- `tables/*.tex`: generated LaTeX tables
- `figures/*.png`: generated manuscript figures
- `build.sh`: end-to-end asset generation and LaTeX build helper

## Build Workflow

From the repository root:

```bash
bash paper/build.sh
```

This does two things:

1. Runs `uv run python paper/scripts/make_paper_assets.py`
2. Attempts a LaTeX build with `latexmk -pdf`, or falls back to `pdflatex` + `bibtex`

If no LaTeX toolchain is installed, the script still generates:

- `outputs/summary_tables/*.csv`
- `paper/tables/*.tex`
- `paper/figures/*.png`

and exits cleanly with a message.

## Asset Generation Notes

The current repository snapshot does not include the full original `outputs/` experiment tree. To keep the paper package reproducible in this checkout, the asset generator uses a locked manuscript-data module in `paper/scripts/manuscript_data.py` to materialize the expected summary CSVs under `outputs/summary_tables/`. It then reads those CSVs back to create the LaTeX tables and figures.

This keeps the package deterministic and lets the paper build from repo root without internet access.

## Regenerating Only Tables and Figures

```bash
uv run python paper/scripts/make_paper_assets.py
```

## Reproducibility Metadata

- Repository: <https://github.com/kmardhani/uncertainty-aware-retfound>
- Commit hash reported in manuscript: `b7e579b183eda54e99d5d3944574fac7dca9d322`
- arXiv categories: `cs.CV`, `cs.LG`

## Caveat

The manuscript is intentionally conservative. It should not be described as a state-of-the-art claim or a clinically validated deployment study. The strongest claim in the paper is about safety/coverage evaluation under dataset shift, not universal superiority of Bayesian or SNGP-style methods.

This work is also an independent research initiative by the author. Although the author is enrolled in the Master of Science in Artificial Intelligence program at the University of Colorado Boulder, the project is not sponsored, supervised, or formally endorsed by the university.
