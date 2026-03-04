#!/usr/bin/env python3
"""
Generate error-analysis figures for the coursework report:
1) Confusion matrix image (final ensemble on official dev)
2) Precision-recall curve (final ensemble probabilities on official dev)
3) Multi-seed comparison chart (selected runs vs ensemble)
4) Metrics CSV/JSON for direct LaTeX table insertion
"""

from __future__ import annotations

import ast
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_MAIN = REPO_ROOT / "data/dontpatronizeme_pcl.tsv"
DATA_DEV_SPLIT = REPO_ROOT / "data/practice splits/dev_semeval_parids-labels.csv"
SUMMARY_JSON = REPO_ROOT / ".bestmodel_runs_ensemble/summary.json"
FIG_DIR = Path(__file__).resolve().parent

MAX_LENGTH = 256
EVAL_BATCH_SIZE = 64
CLEAN_HTML = True


@dataclass
class SelectedRun:
    model_name: str
    seed: int
    weight: float
    weight_dir: Path
    final_dev_f1: float

    @property
    def run_id(self) -> str:
        return f"{self.model_name}__seed_{self.seed}"


def sanitize_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value)


def clean_text(text: str) -> str:
    t = str(text)
    if CLEAN_HTML:
        t = html.unescape(t)
        t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def read_split_labels(split_df: pd.DataFrame) -> np.ndarray:
    label_vec = split_df["label"].apply(ast.literal_eval)
    return label_vec.apply(lambda x: int(any(x))).to_numpy(dtype=np.int64)


def load_dev_data() -> tuple[list[str], np.ndarray]:
    main_df = pd.read_csv(
        DATA_MAIN,
        sep="\t",
        header=None,
        names=["par_id", "article_id", "keyword", "country", "text", "label_0_4"],
    )
    main_df["par_id"] = pd.to_numeric(main_df["par_id"], errors="coerce").astype(int)
    main_df["text"] = main_df["text"].fillna("")

    dev_split = pd.read_csv(DATA_DEV_SPLIT)
    dev_split["par_id"] = pd.to_numeric(dev_split["par_id"], errors="coerce").astype(int)
    dev_split["label_bin"] = read_split_labels(dev_split)

    dev_df = dev_split[["par_id", "label_bin"]].merge(
        main_df[["par_id", "text"]],
        on="par_id",
        how="left",
        sort=False,
        validate="1:1",
    )
    dev_df["text"] = dev_df["text"].astype(str).apply(clean_text)

    return dev_df["text"].tolist(), dev_df["label_bin"].to_numpy(dtype=np.int64)


def load_selected_runs() -> tuple[list[SelectedRun], dict]:
    payload = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))

    final_runs = payload["final_seed_runs"]
    moved = payload.get("moved_model_weights", [])
    moved_lookup = {(m["model_name"], int(m["seed"])): REPO_ROOT / m["path"] for m in moved}

    selected: list[SelectedRun] = []
    for row in final_runs:
        key = (row["model_name"], int(row["seed"]))
        weight_dir = moved_lookup.get(key)
        if weight_dir is None:
            # Fallback to deterministic path
            weight_dir = REPO_ROOT / "BestModel" / "model_weights" / f"{sanitize_name(key[0])}__seed_{key[1]}"

        if not weight_dir.exists():
            raise FileNotFoundError(f"Missing weight directory: {weight_dir}")

        selected.append(
            SelectedRun(
                model_name=key[0],
                seed=key[1],
                weight=float(row["weight"]),
                weight_dir=weight_dir,
                final_dev_f1=float(row["dev_metrics"]["f1_pos"]),
            )
        )
    return selected, payload


def predict_probabilities(weight_dir: Path, texts: list[str]) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(weight_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(weight_dir)
    model.to(device)
    model.eval()

    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )

    ds = torch.utils.data.TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
    )
    loader = DataLoader(ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask = [x.to(device) for x in batch]
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            probs.append(p)

    return np.concatenate(probs, axis=0)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max((tp + tn + fp + fn), 1)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    balanced_acc = (recall + specificity) / 2.0
    auprc = float(average_precision_score(y_true, y_prob))

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "negative_predictive_value": npv,
        "balanced_accuracy": balanced_acc,
        "auprc": auprc,
    }


def plot_confusion_matrix(cm: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1], labels=["Pred 0 (No PCL)", "Pred 1 (PCL)"])
    ax.set_yticks([0, 1], labels=["Ref 0 (No PCL)", "Ref 1 (PCL)"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Reference label")
    ax.set_title("Final Ensemble Confusion Matrix on Dev")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, auprc: float):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    baseline = float((y_true == 1).mean())

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot(recall, precision, linewidth=2.2, label=f"Ensemble PR (AUPRC={auprc:.4f})")
    ax.hlines(
        baseline,
        xmin=0.0,
        xmax=1.0,
        colors="gray",
        linestyles="--",
        linewidth=1.5,
        label=f"Positive-rate baseline ({baseline:.4f})",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve on Dev (Final Ensemble)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_multi_seed_comparison(selected_runs: list[SelectedRun], payload: dict, out_path: Path):
    labels = []
    scores = []

    for r in selected_runs:
        short = "DeBERTa" if "deberta" in r.model_name else "RoBERTa"
        labels.append(f"{short}\nseed={r.seed}")
        scores.append(r.final_dev_f1)

    ens_pre = float(payload["final_ensemble_metrics"]["dev_at_selection_threshold"]["f1_pos"])
    ens_final = float(payload["final_ensemble_metrics"]["dev"]["f1_pos"])
    labels.extend(["Ensemble\npre-retune", "Ensemble\nretuned"])
    scores.extend([ens_pre, ens_final])

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    bars = ax.bar(labels, scores, color=["#4e79a7"] * (len(selected_runs)) + ["#f28e2b", "#e15759"])

    best_single = max(r.final_dev_f1 for r in selected_runs)
    ax.axhline(best_single, linestyle="--", linewidth=1.5, color="#4e79a7", alpha=0.8, label=f"Best single = {best_single:.4f}")
    ax.axhline(ens_final, linestyle="-", linewidth=1.8, color="#e15759", alpha=0.9, label=f"Final ensemble = {ens_final:.4f}")

    for b, s in zip(bars, scores):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.003, f"{s:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0.54, max(scores) + 0.04)
    ax.set_ylabel("Dev F1 (positive class)")
    ax.set_title("Multi-seed / Multi-model Comparison (Selected Runs)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    print("Loading dev data...")
    dev_texts, y_dev = load_dev_data()

    print("Loading selected runs from summary...")
    selected_runs, payload = load_selected_runs()
    weights = np.array([r.weight for r in selected_runs], dtype=np.float64)
    weights = weights / weights.sum()

    print("Running model inference for selected runs...")
    run_probs = []
    for r in selected_runs:
        print(f"  - {r.run_id} from {r.weight_dir}")
        p = predict_probabilities(r.weight_dir, dev_texts)
        run_probs.append(p)

    ensemble_probs = np.average(np.vstack(run_probs), axis=0, weights=weights)

    threshold = float(payload["final_threshold"])
    y_pred = (ensemble_probs >= threshold).astype(np.int64)
    metrics = compute_binary_metrics(y_dev, y_pred, ensemble_probs)

    # Outputs
    confusion_png = FIG_DIR / "error_confusion_matrix_dev.png"
    pr_png = FIG_DIR / "error_precision_recall_curve_dev.png"
    seeds_png = FIG_DIR / "error_multiseed_comparison_dev.png"
    metrics_csv = FIG_DIR / "error_metrics_summary_dev.csv"
    metrics_json = FIG_DIR / "error_metrics_summary_dev.json"

    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=int)
    plot_confusion_matrix(cm, confusion_png)
    plot_pr_curve(y_dev, ensemble_probs, pr_png, metrics["auprc"])
    plot_multi_seed_comparison(selected_runs, payload, seeds_png)

    metrics_rows = [
        ("Threshold", threshold),
        ("TN", metrics["tn"]),
        ("FP", metrics["fp"]),
        ("FN", metrics["fn"]),
        ("TP", metrics["tp"]),
        ("Precision", metrics["precision"]),
        ("Recall", metrics["recall"]),
        ("F1", metrics["f1"]),
        ("Accuracy", metrics["accuracy"]),
        ("Specificity", metrics["specificity"]),
        ("False Positive Rate", metrics["false_positive_rate"]),
        ("False Negative Rate", metrics["false_negative_rate"]),
        ("Negative Predictive Value", metrics["negative_predictive_value"]),
        ("Balanced Accuracy", metrics["balanced_accuracy"]),
        ("AUPRC", metrics["auprc"]),
    ]
    pd.DataFrame(metrics_rows, columns=["metric", "value"]).to_csv(metrics_csv, index=False)

    metrics_payload = {
        "threshold": threshold,
        "metrics": metrics,
        "selected_runs": [
            {
                "model_name": r.model_name,
                "seed": r.seed,
                "weight": r.weight,
                "weight_dir": str(r.weight_dir),
                "final_dev_f1": r.final_dev_f1,
            }
            for r in selected_runs
        ],
        "output_files": {
            "confusion_matrix_png": str(confusion_png),
            "precision_recall_curve_png": str(pr_png),
            "multiseed_comparison_png": str(seeds_png),
            "metrics_csv": str(metrics_csv),
        },
    }
    metrics_json.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("Saved:")
    print(f"  - {confusion_png}")
    print(f"  - {pr_png}")
    print(f"  - {seeds_png}")
    print(f"  - {metrics_csv}")
    print(f"  - {metrics_json}")
    print("Done.")


if __name__ == "__main__":
    main()
