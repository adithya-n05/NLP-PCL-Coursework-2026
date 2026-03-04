#!/usr/bin/env python3
"""
Run quick ablations for coursework Section 5.2 using the selected best runs.

This script:
1) Loads the selected ensemble members from .bestmodel_runs_ensemble/summary.json
2) Retrains each selected model (same model_name + seed) under one-ablation-at-a-time variants
3) Evaluates each model on official dev (positive-class metrics)
4) Builds weighted ensemble metrics per variant
5) Saves CSV outputs for report tables
"""

import argparse
import ast
import html
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_FILE = REPO_ROOT / "data/dontpatronizeme_pcl.tsv"
TRAIN_SPLIT_FILE = REPO_ROOT / "data/practice splits/train_semeval_parids-labels.csv"
DEV_SPLIT_FILE = REPO_ROOT / "data/practice splits/dev_semeval_parids-labels.csv"
SUMMARY_JSON = REPO_ROOT / ".bestmodel_runs_ensemble/summary.json"

DEFAULT_RUNS_DIR = REPO_ROOT / ".ablation_runs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluation"


# ---------------------------------------------------------------------------
# Core constants (match best-model recipe unless ablated)
# ---------------------------------------------------------------------------

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 1
SAVE_TOTAL_LIMIT = 1
LOGGING_STEPS = 50
DATALOADER_NUM_WORKERS = 2
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.005
EXPECTED_DEV_LINES = 2094
NUM_TRAIN_EPOCHS = 4.0
MIXED_PRECISION = "fp16"  # "none", "fp16", "bf16"

TOKENIZERS_PARALLELISM = "false"


@dataclass
class SelectedModel:
    model_name: str
    seed: int
    weight: float
    model_id: str


class WeightedSamplerTrainer(Trainer):
    """Trainer with optional WeightedRandomSampler support."""

    def __init__(self, *args, sample_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Training requires a train_dataset.")
        if self.sample_weights is None:
            return super().get_train_dataloader()

        weights = torch.as_tensor(self.sample_weights, dtype=torch.double)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run 5.2 ablations on selected best runs.")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=SUMMARY_JSON,
        help="Path to run summary containing selected final_seed_runs.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "control",
            "no_weighted_sampler",
            "fixed_threshold_050",
            "max_length_128",
            "no_text_cleanup",
        ],
        choices=[
            "control",
            "no_weighted_sampler",
            "fixed_threshold_050",
            "max_length_128",
            "no_text_cleanup",
        ],
        help="Ablation variants to run.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=(
            "Optional subset of model IDs to run, e.g. "
            "'microsoft_deberta-v3-base__seed_1337 roberta-base__seed_42'."
        ),
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=NUM_TRAIN_EPOCHS,
        help=f"Epochs per run (default from constant: {NUM_TRAIN_EPOCHS}).",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=["none", "fp16", "bf16"],
        default=MIXED_PRECISION,
        help=f"Precision mode (default from constant: {MIXED_PRECISION}).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Directory for temporary checkpoints/logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for ablation CSV outputs.",
    )
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_text(text: str, clean_html_enabled: bool) -> str:
    t = str(text)
    if clean_html_enabled:
        t = html.unescape(t)
        t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def read_split_labels(split_df: pd.DataFrame) -> np.ndarray:
    label_vec = split_df["label"].apply(ast.literal_eval)
    return label_vec.apply(lambda x: int(any(x))).to_numpy(dtype=np.int64)


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_pos": float(p),
        "recall_pos": float(r),
        "f1_pos": float(f1),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def find_best_threshold(y_true: np.ndarray, prob_pos: np.ndarray) -> tuple[float, dict]:
    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + 1e-12, THRESHOLD_STEP)
    best_t = 0.5
    best_metrics = {"f1_pos": -1.0}
    for t in thresholds:
        preds = (prob_pos >= t).astype(int)
        metrics = evaluate_binary(y_true, preds)
        if metrics["f1_pos"] > best_metrics["f1_pos"]:
            best_t = float(t)
            best_metrics = metrics
    return best_t, best_metrics


def to_hf_dataset(df: pd.DataFrame, label_col: str | None = None) -> Dataset:
    data = {"text": df["text"].astype(str).tolist()}
    if label_col is not None:
        data["labels"] = df[label_col].astype(int).tolist()
    return Dataset.from_dict(data)


def build_sample_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=2)
    neg = max(int(counts[0]), 1)
    pos = max(int(counts[1]), 1)
    class_weight = {0: 1.0, 1: float(neg / pos)}
    return np.array([class_weight[int(label)] for label in y], dtype=np.float64)


def normalize_weights(raw: list[float]) -> np.ndarray:
    w = np.array(raw, dtype=np.float64)
    w = np.clip(w, 1e-12, None)
    w /= w.sum()
    return w


def ensemble_probabilities(prob_list: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    if not prob_list:
        raise ValueError("No probability arrays provided for ensembling.")
    stacked = np.vstack([p.astype(np.float64) for p in prob_list])
    return np.average(stacked, axis=0, weights=weights)


def load_data(clean_html_enabled: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    main_df = pd.read_csv(
        MAIN_FILE,
        sep="\t",
        header=None,
        names=["par_id", "article_id", "keyword", "country", "text", "label_0_4"],
    )
    main_df["par_id"] = pd.to_numeric(main_df["par_id"], errors="coerce").astype(int)
    main_df["article_id"] = main_df["article_id"].astype(str)
    main_df["text"] = main_df["text"].fillna("")

    train_split = pd.read_csv(TRAIN_SPLIT_FILE)
    dev_split = pd.read_csv(DEV_SPLIT_FILE)
    for split_df in (train_split, dev_split):
        split_df["par_id"] = pd.to_numeric(split_df["par_id"], errors="coerce").astype(int)
        split_df["label_bin"] = read_split_labels(split_df)

    train_df = train_split[["par_id", "label_bin"]].merge(
        main_df[["par_id", "article_id", "text"]],
        on="par_id",
        how="left",
        sort=False,
        validate="1:1",
    )
    dev_df = dev_split[["par_id", "label_bin"]].merge(
        main_df[["par_id", "article_id", "text"]],
        on="par_id",
        how="left",
        sort=False,
        validate="1:1",
    )

    for df in (train_df, dev_df):
        df["text"] = df["text"].astype(str).apply(lambda x: clean_text(x, clean_html_enabled))

    if len(dev_df) != EXPECTED_DEV_LINES:
        raise ValueError(f"Dev split size mismatch: got {len(dev_df)} expected {EXPECTED_DEV_LINES}")
    if train_df["text"].isna().any() or dev_df["text"].isna().any():
        raise ValueError("Missing text rows after split-id merge.")

    return train_df, dev_df


def load_selected_models(summary_json: Path, model_filter: list[str] | None) -> list[SelectedModel]:
    if not summary_json.exists():
        raise FileNotFoundError(f"Missing summary JSON: {summary_json}")

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    final_seed_runs = payload.get("final_seed_runs", [])
    if not final_seed_runs:
        raise ValueError("No final_seed_runs found in summary JSON.")

    model_filter_set = set(model_filter) if model_filter else None
    selected = []
    for row in final_seed_runs:
        model_name = row["model_name"]
        seed = int(row["seed"])
        weight = float(row["weight"])
        model_id = f"{sanitize_name(model_name)}__seed_{seed}"
        if model_filter_set is not None and model_id not in model_filter_set:
            continue
        selected.append(SelectedModel(model_name=model_name, seed=seed, weight=weight, model_id=model_id))

    if not selected:
        raise ValueError("No models selected after applying --models filter.")
    return selected


def get_precision_flags(mixed_precision: str) -> tuple[bool, bool]:
    if not torch.cuda.is_available() or mixed_precision == "none":
        return False, False
    if mixed_precision == "bf16":
        return True, False
    return False, True


def train_and_predict_dev_probs(
    model_name: str,
    seed: int,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    run_dir: Path,
    *,
    weighted_sampler: bool,
    max_length: int,
    num_train_epochs: float,
    mixed_precision: str,
) -> np.ndarray:
    set_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_ds = to_hf_dataset(train_df, label_col="label_bin").map(tokenize, batched=True, remove_columns=["text"])
    dev_ds = to_hf_dataset(dev_df, label_col="label_bin").map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    bf16, fp16 = get_precision_flags(mixed_precision)

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        evaluation_strategy="no",
        save_strategy="no",
        save_total_limit=SAVE_TOTAL_LIMIT,
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        report_to="none",
        seed=seed,
        data_seed=seed,
        bf16=bf16,
        fp16=fp16,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
    )

    sample_weights = None
    if weighted_sampler:
        sample_weights = build_sample_weights(train_df["label_bin"].to_numpy(dtype=np.int64))

    trainer = WeightedSamplerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        sample_weights=sample_weights,
    )

    trainer.train()
    logits = trainer.predict(dev_ds).predictions
    return torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()


def make_variant_configs() -> dict:
    base = {
        "weighted_sampler": True,
        "max_length": 256,
        "clean_html": True,
        "tune_threshold": True,
        "fixed_threshold": 0.5,
    }
    return {
        "control": dict(base),
        "no_weighted_sampler": {**base, "weighted_sampler": False},
        "fixed_threshold_050": {**base, "tune_threshold": False, "fixed_threshold": 0.5},
        "max_length_128": {**base, "max_length": 128},
        "no_text_cleanup": {**base, "clean_html": False},
    }


def run_variant(
    variant_name: str,
    cfg: dict,
    selected_models: list[SelectedModel],
    data_cache: dict,
    runs_dir: Path,
    num_train_epochs: float,
    mixed_precision: str,
) -> tuple[list[dict], dict]:
    train_df, dev_df = data_cache[cfg["clean_html"]]
    y_dev = dev_df["label_bin"].to_numpy(dtype=np.int64)

    model_rows = []
    probs_for_ensemble = []
    used_models = []

    print("=" * 90)
    print(f"[VARIANT] {variant_name}")
    print(
        "  config:",
        {
            "weighted_sampler": cfg["weighted_sampler"],
            "max_length": cfg["max_length"],
            "clean_html": cfg["clean_html"],
            "tune_threshold": cfg["tune_threshold"],
            "fixed_threshold": cfg["fixed_threshold"],
        },
    )

    for sm in selected_models:
        t0 = time.time()
        run_dir = runs_dir / variant_name / sm.model_id
        print(f"  [RUN] {sm.model_id}")

        prob_dev = train_and_predict_dev_probs(
            model_name=sm.model_name,
            seed=sm.seed,
            train_df=train_df,
            dev_df=dev_df,
            run_dir=run_dir,
            weighted_sampler=cfg["weighted_sampler"],
            max_length=cfg["max_length"],
            num_train_epochs=num_train_epochs,
            mixed_precision=mixed_precision,
        )

        if cfg["tune_threshold"]:
            threshold, metrics = find_best_threshold(y_dev, prob_dev)
            preds = (prob_dev >= threshold).astype(np.int64)
        else:
            threshold = float(cfg["fixed_threshold"])
            preds = (prob_dev >= threshold).astype(np.int64)
            metrics = evaluate_binary(y_dev, preds)

        elapsed = time.time() - t0
        print(
            f"  [DONE] {sm.model_id} "
            f"f1={metrics['f1_pos']:.4f} "
            f"p={metrics['precision_pos']:.4f} "
            f"r={metrics['recall_pos']:.4f} "
            f"thr={threshold:.3f} "
            f"time={elapsed/60:.2f}m"
        )

        row = {
            "variant": variant_name,
            "model_name": sm.model_name,
            "seed": sm.seed,
            "model_id": sm.model_id,
            "weight_from_summary": sm.weight,
            "threshold": threshold,
            "accuracy": metrics["accuracy"],
            "precision_pos": metrics["precision_pos"],
            "recall_pos": metrics["recall_pos"],
            "f1_pos": metrics["f1_pos"],
            "f1_macro": metrics["f1_macro"],
            "runtime_seconds": elapsed,
            "weighted_sampler": cfg["weighted_sampler"],
            "max_length": cfg["max_length"],
            "clean_html": cfg["clean_html"],
            "threshold_mode": "tuned" if cfg["tune_threshold"] else "fixed_0.5",
        }
        model_rows.append(row)
        probs_for_ensemble.append(prob_dev)
        used_models.append(sm)

    if not model_rows:
        raise RuntimeError(f"No model rows produced for variant {variant_name}.")

    ens_weights = normalize_weights([m.weight for m in used_models])
    ens_prob_dev = ensemble_probabilities(probs_for_ensemble, ens_weights)

    if cfg["tune_threshold"]:
        ens_threshold, ens_metrics = find_best_threshold(y_dev, ens_prob_dev)
    else:
        ens_threshold = float(cfg["fixed_threshold"])
        ens_preds = (ens_prob_dev >= ens_threshold).astype(np.int64)
        ens_metrics = evaluate_binary(y_dev, ens_preds)

    best_single = max(model_rows, key=lambda r: r["f1_pos"])
    ensemble_row = {
        "variant": variant_name,
        "n_models": len(used_models),
        "weight_source": "summary_final_seed_runs",
        "threshold": ens_threshold,
        "accuracy": ens_metrics["accuracy"],
        "precision_pos": ens_metrics["precision_pos"],
        "recall_pos": ens_metrics["recall_pos"],
        "f1_pos": ens_metrics["f1_pos"],
        "f1_macro": ens_metrics["f1_macro"],
        "best_single_model_id": best_single["model_id"],
        "best_single_f1_pos": best_single["f1_pos"],
        "ensemble_minus_best_single_f1": ens_metrics["f1_pos"] - best_single["f1_pos"],
        "threshold_mode": "tuned" if cfg["tune_threshold"] else "fixed_0.5",
        "weighted_sampler": cfg["weighted_sampler"],
        "max_length": cfg["max_length"],
        "clean_html": cfg["clean_html"],
    }

    print(
        "  [ENSEMBLE]",
        f"f1={ensemble_row['f1_pos']:.4f}",
        f"best_single={ensemble_row['best_single_f1_pos']:.4f}",
        f"delta={ensemble_row['ensemble_minus_best_single_f1']:+.4f}",
        f"thr={ensemble_row['threshold']:.3f}",
    )
    return model_rows, ensemble_row


def main():
    os = __import__("os")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", TOKENIZERS_PARALLELISM)

    args = parse_args()
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_models = load_selected_models(args.summary_json, args.models)
    variant_cfgs = make_variant_configs()

    print("Selected models:")
    for sm in selected_models:
        print(f"  - {sm.model_id} (weight={sm.weight:.6f})")
    print("Variants:", args.variants)

    # Load both text-cleanup versions once to avoid repeated I/O.
    data_cache = {
        True: load_data(clean_html_enabled=True),
        False: load_data(clean_html_enabled=False),
    }

    all_model_rows = []
    ensemble_rows = []

    wall_t0 = time.time()
    for variant_name in args.variants:
        model_rows, ensemble_row = run_variant(
            variant_name=variant_name,
            cfg=variant_cfgs[variant_name],
            selected_models=selected_models,
            data_cache=data_cache,
            runs_dir=args.runs_dir,
            num_train_epochs=args.num_train_epochs,
            mixed_precision=args.mixed_precision,
        )
        all_model_rows.extend(model_rows)
        ensemble_rows.append(ensemble_row)

    model_df = pd.DataFrame(all_model_rows)
    ensemble_df = pd.DataFrame(ensemble_rows)

    model_csv = args.output_dir / "ablation_model_results.csv"
    ensemble_csv = args.output_dir / "ablation_ensemble_results.csv"
    summary_json = args.output_dir / "ablation_summary.json"

    model_df = model_df.sort_values(by=["variant", "model_name", "seed"]).reset_index(drop=True)
    ensemble_df = ensemble_df.sort_values(by=["variant"]).reset_index(drop=True)

    model_df.to_csv(model_csv, index=False)
    ensemble_df.to_csv(ensemble_csv, index=False)

    payload = {
        "variants": args.variants,
        "selected_models": [
            {
                "model_name": sm.model_name,
                "seed": sm.seed,
                "weight": sm.weight,
                "model_id": sm.model_id,
            }
            for sm in selected_models
        ],
        "num_train_epochs": args.num_train_epochs,
        "mixed_precision": args.mixed_precision,
        "outputs": {
            "model_results_csv": str(model_csv),
            "ensemble_results_csv": str(ensemble_csv),
        },
        "wall_time_seconds": time.time() - wall_t0,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 90)
    print(f"Saved: {model_csv}")
    print(f"Saved: {ensemble_csv}")
    print(f"Saved: {summary_json}")
    print("Done.")


if __name__ == "__main__":
    main()
