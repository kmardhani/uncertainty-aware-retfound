"""Generate summary CSVs, LaTeX tables, and figures for the paper package."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover - exercised at runtime
    raise SystemExit(
        "matplotlib is required to generate paper figures. "
        "Install it in the environment and rerun this script."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from manuscript_data import (  # noqa: E402
    AFFILIATION,
    ARXIV_PRIMARY,
    ARXIV_SECONDARY,
    AUTHOR,
    COMMIT_HASH,
    DATASET_SUMMARY,
    DECISION_POLICY_ROWS,
    EMAIL,
    FEATURE_EXTRACTION_SUMMARY,
    MODEL_COMPARISON_ROWS,
    PROJECT_STATUS,
    REPOSITORY_URL,
    SELECTIVE_REFERRAL_ROWS,
    THRESHOLD_SWEEP_ROWS,
    TITLE,
)


def _repo_root() -> Path:
    return SCRIPT_DIR.parent.parent


def _ensure_dirs() -> dict[str, Path]:
    repo_root = _repo_root()
    paths = {
        "repo_root": repo_root,
        "paper_dir": repo_root / "paper",
        "tables_dir": repo_root / "paper" / "tables",
        "figures_dir": repo_root / "paper" / "figures",
        "summary_dir": repo_root / "outputs" / "summary_tables",
    }
    for path in paths.values():
        if path.suffix == "":
            path.mkdir(parents=True, exist_ok=True)
    return paths


def _write_summary_tables(summary_dir: Path) -> None:
    model_df = pd.DataFrame(MODEL_COMPARISON_ROWS)
    threshold_df = pd.DataFrame(THRESHOLD_SWEEP_ROWS)
    selective_df = pd.DataFrame(SELECTIVE_REFERRAL_ROWS)
    decision_df = pd.DataFrame(DECISION_POLICY_ROWS)

    model_df.to_csv(summary_dir / "model_comparison.csv", index=False)
    threshold_df.to_csv(summary_dir / "threshold_sweep_summary.csv", index=False)
    selective_df.to_csv(summary_dir / "selective_referral_summary.csv", index=False)
    decision_df.to_csv(summary_dir / "decision_policy_comparison.csv", index=False)

    payload = {
        "model_comparison": model_df.to_dict(orient="records"),
        "threshold_sweep_summary": threshold_df.to_dict(orient="records"),
        "selective_referral_summary": selective_df.to_dict(orient="records"),
        "decision_policy_comparison": decision_df.to_dict(orient="records"),
    }
    (summary_dir / "summary_tables.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _read_summary_tables(summary_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "model": pd.read_csv(summary_dir / "model_comparison.csv"),
        "threshold": pd.read_csv(summary_dir / "threshold_sweep_summary.csv"),
        "selective": pd.read_csv(summary_dir / "selective_referral_summary.csv"),
        "decision": pd.read_csv(summary_dir / "decision_policy_comparison.csv"),
    }


def _escape_latex(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _format_value(value: object, decimals: int = 4) -> str:
    if value is None or pd.isna(value):
        return "--"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, int) or float(value).is_integer():
            return str(int(value))
        return f"{float(value):.{decimals}f}"
    return _escape_latex(value)


def _write_table(
    *,
    path: Path,
    dataframe: pd.DataFrame,
    columns: list[str],
    headers: list[str],
    caption: str,
    label: str,
    decimals: dict[str, int] | None = None,
    font_size: str = r"\small",
    column_spec: str | None = None,
    tabular_environment: str = "tabular",
    table_width: str = r"\textwidth",
    preformatted_columns: set[str] | None = None,
) -> None:
    decimals = decimals or {}
    preformatted_columns = preformatted_columns or set()
    align = column_spec or ("l" + "r" * (len(columns) - 1))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        font_size,
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
    ]
    if tabular_environment == "tabularx":
        lines.append(rf"\begin{{tabularx}}{{{table_width}}}{{{align}}}")
    else:
        lines.append(rf"\begin{{{tabular_environment}}}{{{align}}}")
    lines.extend(
        [
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
        ]
    )
    for record in dataframe[columns].to_dict(orient="records"):
        row = []
        for column in columns:
            if column in preformatted_columns:
                row.append(str(record[column]))
            else:
                row.append(_format_value(record[column], decimals.get(column, 4)))
        lines.append(" & ".join(row) + r" \\")
    end_environment = "tabularx" if tabular_environment == "tabularx" else tabular_environment
    lines.extend([r"\bottomrule", rf"\end{{{end_environment}}}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _dataset_summary_table(paths: dict[str, Path]) -> None:
    dataframe = pd.DataFrame(DATASET_SUMMARY)
    _write_table(
        path=paths["tables_dir"] / "dataset_summary.tex",
        dataframe=dataframe,
        columns=[
            "dataset",
            "total_images",
            "non_referable",
            "referable",
            "train_count",
            "val_count",
            "test_count",
        ],
        headers=["Dataset", "Total", "Non-ref", "Ref", "Train", "Val", "Test"],
        caption="Dataset sizes after binary referable diabetic retinopathy mapping.",
        label="tab:dataset-summary",
        decimals={"total_images": 0, "non_referable": 0, "referable": 0, "train_count": 0, "val_count": 0, "test_count": 0},
    )


def _feature_summary_table(paths: dict[str, Path]) -> None:
    dataframe = pd.DataFrame(FEATURE_EXTRACTION_SUMMARY)
    _write_table(
        path=paths["tables_dir"] / "feature_extraction_summary.tex",
        dataframe=dataframe,
        columns=["dataset", "backbone", "checkpoint", "feature_dim", "resize", "center_crop", "device", "batch_size"],
        headers=["Dataset", "Backbone", "Checkpoint", "Dim", "Resize", "Crop", "Device", "Batch"],
        caption="Frozen RETFound feature extraction settings used for cached-feature experiments.",
        label="tab:feature-summary",
        font_size=r"\footnotesize",
        column_spec=r"l l X r r r l r",
        tabular_environment="tabularx",
        decimals={"feature_dim": 0, "resize": 0, "center_crop": 0, "batch_size": 0},
    )


def _model_label(model_name: str) -> str:
    mapping = {
        "cached_softmax": "Softmax",
        "cached_softmax_temperature_scaled": "Softmax+Temp",
        "variational_bayesian_val_loss_selected": "Bayes (val-loss)",
        "variational_bayesian_max_sensitivity_sweep": "Bayes (max-sens)",
        "variational_bayesian_balanced_sweep": "Bayes (balanced)",
        "laplace_val_loss_selected": "Laplace (val-loss)",
        "laplace_sensitivity_selected": "Laplace (sens)",
        "sngp_val_loss_selected": "SNGP (val-loss)",
        "sngp_sensitivity_selected": "SNGP (sens)",
        "ddr_softmax": "DDR Softmax",
        "ddr_bayesian_val_loss_selected": "DDR Bayes (val-loss)",
        "ddr_bayesian_sensitivity_selected": "DDR Bayes (sens)",
        "ddr_from_aptos_sngp_sensitivity": "APTOS->DDR SNGP",
    }
    return mapping.get(model_name, model_name)


def _aptos_model_table(paths: dict[str, Path], model_df: pd.DataFrame) -> None:
    subset = model_df[model_df["dataset"] == "APTOS"].copy()
    subset["display_model"] = subset["model"].map(_model_label)
    _write_table(
        path=paths["tables_dir"] / "aptos_model_comparison.tex",
        dataframe=subset,
        columns=[
            "display_model",
            "accuracy",
            "auc",
            "sensitivity",
            "specificity",
            "balanced_accuracy",
            "ece",
            "false_negatives",
            "false_positives",
        ],
        headers=["Model", "Acc", "AUC", "Sens", "Spec", "Bal Acc", "ECE", "FN", "FP"],
        caption="APTOS validation comparison for full-coverage cached-feature heads.",
        label="tab:aptos-models",
        decimals={
            "accuracy": 4,
            "auc": 4,
            "sensitivity": 4,
            "specificity": 4,
            "balanced_accuracy": 4,
            "ece": 4,
            "false_negatives": 0,
            "false_positives": 0,
        },
    )


def _ddr_model_table(paths: dict[str, Path], model_df: pd.DataFrame) -> None:
    subset = model_df[model_df["dataset"] == "DDR"].copy()
    subset["display_model"] = subset["model"].map(_model_label)
    _write_table(
        path=paths["tables_dir"] / "ddr_model_comparison.tex",
        dataframe=subset,
        columns=[
            "display_model",
            "accuracy",
            "auc",
            "sensitivity",
            "specificity",
            "balanced_accuracy",
            "ece",
            "false_negatives",
            "false_positives",
        ],
        headers=["Model", "Acc", "AUC", "Sens", "Spec", "Bal Acc", "ECE", "FN", "FP"],
        caption="DDR validation comparison for full-coverage native DDR heads and APTOS-to-DDR SNGP transfer.",
        label="tab:ddr-models",
        decimals={
            "accuracy": 4,
            "auc": 4,
            "sensitivity": 4,
            "specificity": 4,
            "balanced_accuracy": 4,
            "ece": 4,
            "false_negatives": 0,
            "false_positives": 0,
        },
    )


def _ddr_threshold_table(paths: dict[str, Path], threshold_df: pd.DataFrame) -> None:
    subset = threshold_df[threshold_df["sweep_name"].str.startswith("ddr")].copy()
    subset["sweep"] = subset["sweep_name"].replace(
        {
            "ddr_softmax_temp": "Softmax+Temp",
            "ddr_bayes_val_loss": "Bayes (val-loss)",
            "ddr_bayes_sensitivity": "Bayes (sens)",
            "ddr_from_aptos_sngp": "APTOS->DDR SNGP",
        }
    )
    subset["policy"] = subset["selected_policy"].replace(
        {
            "best_balanced_accuracy": "Best bal acc",
            "lowest_false_negatives": "Lowest FN",
        }
    )
    _write_table(
        path=paths["tables_dir"] / "ddr_threshold_summary.tex",
        dataframe=subset,
        columns=[
            "sweep",
            "policy",
            "threshold",
            "sensitivity",
            "specificity",
            "balanced_accuracy",
            "false_negatives",
            "false_positives",
        ],
        headers=["Sweep", "Policy", "Thr", "Sens", "Spec", "Bal Acc", "FN", "FP"],
        caption="DDR threshold-sweep summary for balanced and low-false-negative operating points.",
        label="tab:ddr-thresholds",
        decimals={
            "threshold": 2,
            "sensitivity": 4,
            "specificity": 4,
            "balanced_accuracy": 4,
            "false_negatives": 0,
            "false_positives": 0,
        },
    )


def _ddr_selective_table(paths: dict[str, Path], selective_df: pd.DataFrame) -> None:
    ddr_rows = selective_df[
        selective_df["run_name"].isin(
            [
                "ddr_bayesian_sensitivity_confidence",
                "ddr_bayesian_sensitivity_predictive_entropy",
                "ddr_bayesian_sensitivity_probability_variance",
                "ddr_bayesian_sensitivity_mutual_information",
                "ddr_from_aptos_sngp_sensitivity_entropy",
                "ddr_from_aptos_sngp_sensitivity_variance",
                "ddr_from_aptos_sngp_sensitivity_sngp_uncertainty",
            ]
        )
    ].copy()
    ddr_rows["signal"] = ddr_rows["run_name"].replace(
        {
            "ddr_bayesian_sensitivity_confidence": "DDR Bayes confidence",
            "ddr_bayesian_sensitivity_predictive_entropy": "DDR Bayes entropy",
            "ddr_bayesian_sensitivity_probability_variance": "DDR Bayes prob var",
            "ddr_bayesian_sensitivity_mutual_information": "DDR Bayes MI",
            "ddr_from_aptos_sngp_sensitivity_entropy": "APTOS->DDR SNGP entropy",
            "ddr_from_aptos_sngp_sensitivity_variance": "APTOS->DDR SNGP variance",
            "ddr_from_aptos_sngp_sensitivity_sngp_uncertainty": "APTOS->DDR SNGP combined",
        }
    )
    _write_table(
        path=paths["tables_dir"] / "ddr_selective_referral_80.tex",
        dataframe=ddr_rows,
        columns=[
            "signal",
            "coverage",
            "referral_rate",
            "sensitivity",
            "specificity",
            "balanced_accuracy",
            "false_negatives",
            "false_positives",
        ],
        headers=["Signal", "Coverage", "Referral", "Sens", "Spec", "Bal Acc", "FN", "FP"],
        caption="DDR selective-referral comparison at approximately 80\\% accepted coverage.",
        label="tab:ddr-selective",
        font_size=r"\footnotesize",
        column_spec=r"X r r r r r r r",
        tabular_environment="tabularx",
        decimals={
            "coverage": 4,
            "referral_rate": 4,
            "sensitivity": 4,
            "specificity": 4,
            "balanced_accuracy": 4,
            "false_negatives": 0,
            "false_positives": 0,
        },
    )


def _sngp_transfer_table(paths: dict[str, Path], model_df: pd.DataFrame) -> None:
    subset = model_df[
        model_df["model"].isin(["sngp_sensitivity_selected", "ddr_from_aptos_sngp_sensitivity"])
    ].copy()
    subset["setting"] = subset["model"].replace(
        {
            "sngp_sensitivity_selected": "APTOS internal",
            "ddr_from_aptos_sngp_sensitivity": "APTOS->DDR transfer",
        }
    )
    _write_table(
        path=paths["tables_dir"] / "sngp_internal_vs_transfer.tex",
        dataframe=subset,
        columns=[
            "setting",
            "accuracy",
            "auc",
            "sensitivity",
            "specificity",
            "balanced_accuracy",
            "ece",
            "false_negatives",
            "false_positives",
        ],
        headers=["Setting", "Acc", "AUC", "Sens", "Spec", "Bal Acc", "ECE", "FN", "FP"],
        caption="SNGP internal APTOS validation versus APTOS-to-DDR transfer evaluation.",
        label="tab:sngp-transfer",
        decimals={
            "accuracy": 4,
            "auc": 4,
            "sensitivity": 4,
            "specificity": 4,
            "balanced_accuracy": 4,
            "ece": 4,
            "false_negatives": 0,
            "false_positives": 0,
        },
    )


def _reproducibility_table(paths: dict[str, Path]) -> None:
    dataframe = pd.DataFrame(
        [
            {"item": "Repository", "value": REPOSITORY_URL},
            {"item": "Commit hash", "value": COMMIT_HASH},
            {"item": "Author", "value": AUTHOR},
            {"item": "Affiliation", "value": AFFILIATION.replace("\\\\", "; ")},
            {"item": "Email", "value": EMAIL},
            {"item": "Title", "value": TITLE},
            {"item": "arXiv primary", "value": ARXIV_PRIMARY},
            {"item": "arXiv secondary", "value": ARXIV_SECONDARY},
            {"item": "Project status", "value": PROJECT_STATUS},
            {"item": "Task", "value": "Binary referable diabetic retinopathy"},
            {"item": "Backbone", "value": "Frozen RETFound_mae"},
            {"item": "Feature dimension", "value": "1024"},
            {"item": "APTOS split", "value": "2930 / 366 / 366"},
            {"item": "DDR split", "value": "8765 / 1878 / 1879"},
        ]
    )
    dataframe["rendered_value"] = dataframe.apply(_format_reproducibility_value, axis=1)
    _write_table(
        path=paths["tables_dir"] / "reproducibility_summary.tex",
        dataframe=dataframe,
        columns=["item", "rendered_value"],
        headers=["Item", "Value"],
        caption="Reproducibility metadata for the manuscript package.",
        label="tab:reproducibility",
        font_size=r"\footnotesize",
        column_spec=r"p{0.23\textwidth}X",
        tabular_environment="tabularx",
        preformatted_columns={"rendered_value"},
    )


def _format_reproducibility_value(row: pd.Series) -> str:
    item = str(row["item"])
    value = str(row["value"])
    if item == "Repository":
        return rf"\url{{{value}}}"
    if item in {"Commit hash", "Email"}:
        return rf"\nolinkurl{{{value}}}"
    return _escape_latex(value)


def _write_all_tables(paths: dict[str, Path], tables: dict[str, pd.DataFrame]) -> None:
    _dataset_summary_table(paths)
    _feature_summary_table(paths)
    _aptos_model_table(paths, tables["model"])
    _ddr_model_table(paths, tables["model"])
    _ddr_threshold_table(paths, tables["threshold"])
    _ddr_selective_table(paths, tables["selective"])
    _sngp_transfer_table(paths, tables["model"])
    _reproducibility_table(paths)


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def _make_aptos_full_coverage_figure(paths: dict[str, Path], model_df: pd.DataFrame) -> None:
    subset = model_df[
        model_df["model"].isin(
            [
                "cached_softmax",
                "variational_bayesian_max_sensitivity_sweep",
                "laplace_sensitivity_selected",
                "sngp_sensitivity_selected",
            ]
        )
    ].copy()
    labels = ["Softmax", "Bayes sens", "Laplace sens", "SNGP sens"]
    x = list(range(len(subset)))
    width = 0.35
    plt.figure(figsize=(7.5, 4.5))
    plt.bar([value - width / 2 for value in x], subset["false_negatives"], width=width, label="False negatives")
    plt.bar([value + width / 2 for value in x], subset["false_positives"], width=width, label="False positives")
    plt.xticks(x, labels, rotation=10)
    plt.ylabel("Count")
    plt.title("APTOS full-coverage error counts")
    plt.legend()
    _save_figure(paths["figures_dir"] / "aptos_full_coverage_comparison.png")


def _make_ddr_full_coverage_figure(paths: dict[str, Path], model_df: pd.DataFrame) -> None:
    subset = model_df[
        model_df["model"].isin(
            [
                "ddr_softmax",
                "ddr_bayesian_val_loss_selected",
                "ddr_bayesian_sensitivity_selected",
                "ddr_from_aptos_sngp_sensitivity",
            ]
        )
    ].copy()
    labels = ["DDR softmax", "DDR Bayes val", "DDR Bayes sens", "APTOS->DDR SNGP"]
    x = list(range(len(subset)))
    width = 0.35
    plt.figure(figsize=(8.2, 4.8))
    plt.bar([value - width / 2 for value in x], subset["false_negatives"], width=width, label="False negatives")
    plt.bar([value + width / 2 for value in x], subset["false_positives"], width=width, label="False positives")
    plt.xticks(x, labels, rotation=12)
    plt.ylabel("Count")
    plt.title("DDR full-coverage error counts")
    plt.legend()
    _save_figure(paths["figures_dir"] / "ddr_full_coverage_comparison.png")


def _make_ddr_sens_spec_figure(paths: dict[str, Path], model_df: pd.DataFrame) -> None:
    subset = model_df[
        model_df["model"].isin(
            [
                "ddr_softmax",
                "ddr_bayesian_val_loss_selected",
                "ddr_bayesian_sensitivity_selected",
                "ddr_from_aptos_sngp_sensitivity",
            ]
        )
    ].copy()
    labels = ["DDR softmax", "DDR Bayes val", "DDR Bayes sens", "APTOS->DDR SNGP"]
    x = list(range(len(subset)))
    width = 0.35
    plt.figure(figsize=(8.2, 4.8))
    plt.bar([value - width / 2 for value in x], subset["sensitivity"], width=width, label="Sensitivity")
    plt.bar([value + width / 2 for value in x], subset["specificity"], width=width, label="Specificity")
    plt.xticks(x, labels, rotation=12)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Rate")
    plt.title("DDR sensitivity and specificity")
    plt.legend()
    _save_figure(paths["figures_dir"] / "ddr_sensitivity_specificity.png")


def _make_selective_referral_figure(paths: dict[str, Path], selective_df: pd.DataFrame) -> None:
    subset = selective_df[
        selective_df["run_name"].isin(
            [
                "ddr_bayesian_sensitivity_confidence",
                "ddr_bayesian_sensitivity_predictive_entropy",
                "ddr_bayesian_sensitivity_probability_variance",
                "ddr_bayesian_sensitivity_mutual_information",
                "ddr_from_aptos_sngp_sensitivity_entropy",
                "ddr_from_aptos_sngp_sensitivity_variance",
                "ddr_from_aptos_sngp_sensitivity_sngp_uncertainty",
            ]
        )
    ].copy()
    subset["label"] = subset["run_name"].replace(
        {
            "ddr_bayesian_sensitivity_confidence": "Bayes conf",
            "ddr_bayesian_sensitivity_predictive_entropy": "Bayes entropy",
            "ddr_bayesian_sensitivity_probability_variance": "Bayes prob var",
            "ddr_bayesian_sensitivity_mutual_information": "Bayes MI",
            "ddr_from_aptos_sngp_sensitivity_entropy": "SNGP entropy",
            "ddr_from_aptos_sngp_sensitivity_variance": "SNGP variance",
            "ddr_from_aptos_sngp_sensitivity_sngp_uncertainty": "SNGP combined",
        }
    )
    plt.figure(figsize=(9.0, 4.8))
    plt.bar(subset["label"], subset["false_negatives"], color="#4C78A8")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Accepted-case false negatives")
    plt.title("Selective referral at approximately 80% coverage on DDR")
    _save_figure(paths["figures_dir"] / "ddr_selective_referral_80.png")


def _make_threshold_summary_figure(paths: dict[str, Path], threshold_df: pd.DataFrame) -> None:
    subset = threshold_df[threshold_df["sweep_name"].str.startswith("ddr")].copy()
    display_order = [
        "ddr_softmax_temp",
        "ddr_bayes_val_loss",
        "ddr_bayes_sensitivity",
        "ddr_from_aptos_sngp",
    ]
    labels = ["Softmax+Temp", "Bayes val", "Bayes sens", "APTOS->DDR SNGP"]
    x = list(range(len(display_order)))
    width = 0.35
    best_balanced = []
    lowest_fn = []
    for name in display_order:
        best_row = subset[(subset["sweep_name"] == name) & (subset["selected_policy"] == "best_balanced_accuracy")]
        low_row = subset[(subset["sweep_name"] == name) & (subset["selected_policy"] == "lowest_false_negatives")]
        best_balanced.append(float(best_row.iloc[0]["false_negatives"]) if not best_row.empty else float("nan"))
        lowest_fn.append(float(low_row.iloc[0]["false_negatives"]) if not low_row.empty else float("nan"))
    plt.figure(figsize=(8.4, 4.8))
    plt.bar([value - width / 2 for value in x], best_balanced, width=width, label="Best balanced accuracy")
    plt.bar([value + width / 2 for value in x], lowest_fn, width=width, label="Lowest false negatives")
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("False negatives")
    plt.title("DDR threshold-sweep policy comparison")
    plt.legend()
    _save_figure(paths["figures_dir"] / "ddr_threshold_sweep_summary.png")


def _make_sngp_transfer_figure(paths: dict[str, Path], model_df: pd.DataFrame) -> None:
    subset = model_df[
        model_df["model"].isin(["sngp_sensitivity_selected", "ddr_from_aptos_sngp_sensitivity"])
    ].copy()
    subset.index = ["APTOS internal", "APTOS->DDR transfer"]
    metrics = ["sensitivity", "specificity", "balanced_accuracy", "auc"]
    x = list(range(len(metrics)))
    width = 0.35
    plt.figure(figsize=(8.0, 4.8))
    plt.bar(
        [value - width / 2 for value in x],
        [float(subset.loc["APTOS internal", metric]) for metric in metrics],
        width=width,
        label="APTOS internal",
    )
    plt.bar(
        [value + width / 2 for value in x],
        [float(subset.loc["APTOS->DDR transfer", metric]) for metric in metrics],
        width=width,
        label="APTOS->DDR transfer",
    )
    plt.xticks(x, ["Sensitivity", "Specificity", "Balanced acc", "AUC"], rotation=10)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("SNGP internal validation versus transfer")
    plt.legend()
    _save_figure(paths["figures_dir"] / "sngp_internal_vs_transfer.png")


def _write_all_figures(paths: dict[str, Path], tables: dict[str, pd.DataFrame]) -> None:
    plt.style.use("default")
    _make_aptos_full_coverage_figure(paths, tables["model"])
    _make_ddr_full_coverage_figure(paths, tables["model"])
    _make_ddr_sens_spec_figure(paths, tables["model"])
    _make_selective_referral_figure(paths, tables["selective"])
    _make_threshold_summary_figure(paths, tables["threshold"])
    _make_sngp_transfer_figure(paths, tables["model"])


def main() -> None:
    paths = _ensure_dirs()
    _write_summary_tables(paths["summary_dir"])
    tables = _read_summary_tables(paths["summary_dir"])
    _write_all_tables(paths, tables)
    _write_all_figures(paths, tables)
    print(f"Wrote summary tables to: {paths['summary_dir']}")
    print(f"Wrote LaTeX tables to: {paths['tables_dir']}")
    print(f"Wrote figures to: {paths['figures_dir']}")


if __name__ == "__main__":
    main()
