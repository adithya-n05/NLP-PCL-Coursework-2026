#!/usr/bin/env python3
import ast
import html
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# -----------------------------------------------------------------------------
# User constants
# -----------------------------------------------------------------------------

MODEL_NAMES = ["roberta-base", "microsoft/deberta-v3-base"]
SEEDS = [42, 1337, 627345]
TOP_K = 4
ENSEMBLE_WEIGHTING = "val_f1"  # "val_f1" or "uniform"
LOCAL_TEST_LABELS_FILE = Path("resources/PCL_testlabels_binary.txt")  # Set to None to disable

MAX_LENGTH = 256
NUM_TRAIN_EPOCHS = 4
MIXED_PRECISION = "none"  # "bf16", "fp16", "none"


# -----------------------------------------------------------------------------
# Fixed data and output paths
# -----------------------------------------------------------------------------

MAIN_FILE = Path("data/dontpatronizeme_pcl.tsv")
TRAIN_SPLIT_FILE = Path("data/practice splits/train_semeval_parids-labels.csv")
DEV_SPLIT_FILE = Path("data/practice splits/dev_semeval_parids-labels.csv")
TEST_FILE = Path("data/task4_test.tsv")

OUTPUT_DIR = Path("BestModel")
RUNS_DIR = Path(".bestmodel_runs_ensemble")
SUMMARY_JSON = RUNS_DIR / "summary.json"
MODEL_WEIGHTS_DIR = OUTPUT_DIR / "model_weights"


# -----------------------------------------------------------------------------
# Fixed pipeline recipe
# -----------------------------------------------------------------------------

INTERNAL_FOLDS = 5
SPLIT_SEED = 42

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 1
EARLY_STOPPING_PATIENCE = 2
SAVE_TOTAL_LIMIT = 1
LOGGING_STEPS = 50
DATALOADER_NUM_WORKERS = 2
USE_WEIGHTED_SAMPLER = True

THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.005

EXPECTED_DEV_LINES = 2094
EXPECTED_TEST_LINES = 3832
CLEAN_HTML = True

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.dont_write_bytecode = True


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sanitize_name(value):
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value)


def clean_text(text):
    t = str(text)
    if CLEAN_HTML:
        t = html.unescape(t)
        t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def read_split_labels(split_df):
    label_vec = split_df["label"].apply(ast.literal_eval)
    return label_vec.apply(lambda x: int(any(x))).to_numpy(dtype=np.int64)


def read_binary_labels(path):
    values = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        s = raw.strip()
        if not s:
            continue
        if s not in {"0", "1"}:
            raise ValueError(f"Non-binary label at line {i}: {s!r}")
        values.append(int(s))
    return np.array(values, dtype=np.int64)


def load_data():
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

    test_df = pd.read_csv(
        TEST_FILE,
        sep="\t",
        header=None,
        names=["row_id", "article_id", "keyword", "country", "text"],
    )
    test_df["article_id"] = test_df["article_id"].astype(str)
    test_df["text"] = test_df["text"].fillna("")

    for df in (train_df, dev_df, test_df):
        df["text"] = df["text"].astype(str).apply(clean_text)

    if train_df["text"].isna().any() or dev_df["text"].isna().any():
        raise ValueError("Missing text rows after split-id merge.")

    if len(dev_df) != EXPECTED_DEV_LINES:
        raise ValueError(f"Dev split size mismatch: got {len(dev_df)} expected {EXPECTED_DEV_LINES}")
    if len(test_df) != EXPECTED_TEST_LINES:
        raise ValueError(f"Test split size mismatch: got {len(test_df)} expected {EXPECTED_TEST_LINES}")

    return train_df, dev_df, test_df


def build_internal_split(train_df):
    y = train_df["label_bin"].to_numpy(dtype=np.int64)
    groups = train_df["article_id"].astype(str).to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=INTERNAL_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    tr_idx, va_idx = next(sgkf.split(np.zeros(len(train_df)), y, groups))

    train_inner = train_df.iloc[tr_idx].reset_index(drop=True)
    val_inner = train_df.iloc[va_idx].reset_index(drop=True)
    return train_inner, val_inner


def evaluate_binary(y_true, y_pred):
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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return evaluate_binary(labels.astype(int), preds.astype(int))


def find_best_threshold(y_true, prob_pos):
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


def to_hf_dataset(df, label_col=None):
    data = {"text": df["text"].astype(str).tolist()}
    if label_col is not None:
        data["labels"] = df[label_col].astype(int).tolist()
    return Dataset.from_dict(data)


def build_sample_weights(y):
    counts = np.bincount(y.astype(int), minlength=2)
    neg = max(int(counts[0]), 1)
    pos = max(int(counts[1]), 1)
    class_weight = {0: 1.0, 1: float(neg / pos)}
    return np.array([class_weight[int(label)] for label in y], dtype=np.float64)


def get_precision_flags():
    if not torch.cuda.is_available() or MIXED_PRECISION == "none":
        return False, False
    if MIXED_PRECISION == "bf16":
        return True, False
    return False, True


def train_and_predict(model_name, seed, train_df, eval_df, predict_dfs, run_dir, evaluate_during_training, save_model_artifacts=False):
    set_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_ds = to_hf_dataset(train_df, label_col="label_bin").map(tokenize, batched=True, remove_columns=["text"])

    eval_ds = None
    if evaluate_during_training and eval_df is not None:
        eval_ds = to_hf_dataset(eval_df, label_col="label_bin").map(tokenize, batched=True, remove_columns=["text"])

    pred_ds = {}
    for name, df in predict_dfs.items():
        pred_ds[name] = to_hf_dataset(df).map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    bf16, fp16 = get_precision_flags()

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        evaluation_strategy="epoch" if evaluate_during_training else "no",
        save_strategy="epoch" if evaluate_during_training else "no",
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=evaluate_during_training,
        metric_for_best_model="f1_pos",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        report_to="none",
        seed=seed,
        data_seed=seed,
        bf16=bf16,
        fp16=fp16,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
    )

    callbacks = []
    if evaluate_during_training and EARLY_STOPPING_PATIENCE > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE))

    sample_weights = None
    if USE_WEIGHTED_SAMPLER:
        sample_weights = build_sample_weights(train_df["label_bin"].to_numpy(dtype=np.int64))

    trainer = WeightedSamplerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics if evaluate_during_training else None,
        callbacks=callbacks,
        sample_weights=sample_weights,
    )

    trainer.train()
    raw_eval = trainer.evaluate() if evaluate_during_training and eval_ds is not None else {}

    if save_model_artifacts:
        export_dir = run_dir / "final_model"
        export_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(export_dir))
        tokenizer.save_pretrained(str(export_dir))

    probs = {}
    for name, ds in pred_ds.items():
        logits = trainer.predict(ds).predictions
        probs[name] = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    clean_eval = {
        "eval_loss": float(raw_eval.get("eval_loss", 0.0)) if raw_eval else 0.0,
        "accuracy": float(raw_eval.get("eval_accuracy", 0.0)) if raw_eval else 0.0,
        "precision_pos": float(raw_eval.get("eval_precision_pos", 0.0)) if raw_eval else 0.0,
        "recall_pos": float(raw_eval.get("eval_recall_pos", 0.0)) if raw_eval else 0.0,
        "f1_pos": float(raw_eval.get("eval_f1_pos", 0.0)) if raw_eval else 0.0,
        "f1_macro": float(raw_eval.get("eval_f1_macro", 0.0)) if raw_eval else 0.0,
    }
    return clean_eval, probs


def normalize_weights(raw):
    w = np.array(raw, dtype=np.float64)
    w = np.clip(w, 1e-12, None)
    w /= w.sum()
    return w


def ensemble_probabilities(prob_list, weights):
    if not prob_list:
        raise ValueError("No probability arrays provided for ensembling.")
    stacked = np.vstack([p.astype(np.float64) for p in prob_list])
    return np.average(stacked, axis=0, weights=weights)


def write_prediction_file(preds, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(int(v)) for v in preds.tolist()) + "\n", encoding="utf-8")


def move_selected_models_to_bestmodel(final_runs):
    MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    moved = []
    for run in final_runs:
        src = run["run_dir"] / "final_model"
        dst = MODEL_WEIGHTS_DIR / f"{sanitize_name(run['model_name'])}__seed_{run['seed']}"
        if not src.exists():
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
        moved.append({"model_name": run["model_name"], "seed": str(run["seed"]), "path": str(dst)})
    return moved


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)

    train_df, dev_df, test_df = load_data()
    y_dev = dev_df["label_bin"].to_numpy(dtype=np.int64)
    y_test = None
    if LOCAL_TEST_LABELS_FILE is not None:
        labels_path = Path(LOCAL_TEST_LABELS_FILE)
        y_test = read_binary_labels(labels_path)
        if len(y_test) != len(test_df):
            raise ValueError(
                f"Local test labels mismatch: got {len(y_test)} expected {len(test_df)}"
            )
        print(f"Local test eval enabled with labels: {labels_path}")

    print(f"Loaded: train={len(train_df)} dev={len(dev_df)} test={len(test_df)}")
    print(f"Models: {MODEL_NAMES}")
    print(f"Seeds: {SEEDS}")

    inner_train_df, inner_val_df = build_internal_split(train_df)
    y_val = inner_val_df["label_bin"].to_numpy(dtype=np.int64)

    scout_runs = []
    for model_name in MODEL_NAMES:
        for seed in SEEDS:
            run_dir = RUNS_DIR / "scout" / sanitize_name(model_name) / f"seed_{seed}"
            print("=" * 80)
            print(f"[SCOUT] model={model_name} seed={seed}")
            _, probs = train_and_predict(
                model_name=model_name,
                seed=seed,
                train_df=inner_train_df,
                eval_df=inner_val_df,
                predict_dfs={"val": inner_val_df, "dev": dev_df, "test": test_df},
                run_dir=run_dir,
                evaluate_during_training=True,
                save_model_artifacts=False,
            )

            tuned_threshold, val_metrics = find_best_threshold(y_val, probs["val"])
            dev_pred = (probs["dev"] >= tuned_threshold).astype(np.int64)
            dev_metrics = evaluate_binary(y_dev, dev_pred)
            local_test_metrics = None
            if y_test is not None:
                test_pred = (probs["test"] >= tuned_threshold).astype(np.int64)
                local_test_metrics = evaluate_binary(y_test, test_pred)

            scout_runs.append(
                {
                    "model_name": model_name,
                    "seed": seed,
                    "tuned_threshold": tuned_threshold,
                    "val_metrics": val_metrics,
                    "dev_metrics": dev_metrics,
                    "local_test_metrics": local_test_metrics,
                    "val_probs": probs["val"],
                    "dev_probs": probs["dev"],
                    "test_probs": probs["test"],
                }
            )

            scout_line = (
                f"[SCOUT DONE] model={model_name} seed={seed} "
                f"val_f1={val_metrics['f1_pos']:.4f} dev_f1={dev_metrics['f1_pos']:.4f} "
                f"thr={tuned_threshold:.3f}"
            )
            if local_test_metrics is not None:
                scout_line += f" local_test_f1={local_test_metrics['f1_pos']:.4f}"
            print(scout_line)

    scout_runs = sorted(scout_runs, key=lambda r: r["val_metrics"]["f1_pos"], reverse=True)
    selected = scout_runs[: min(TOP_K, len(scout_runs))]
    if not selected:
        raise RuntimeError("No scout runs completed.")

    if ENSEMBLE_WEIGHTING == "val_f1":
        scout_weights_raw = [r["val_metrics"]["f1_pos"] for r in selected]
    else:
        scout_weights_raw = [1.0 for _ in selected]
    scout_weights = normalize_weights(scout_weights_raw)

    scout_val_probs = ensemble_probabilities([r["val_probs"] for r in selected], scout_weights)
    scout_dev_probs = ensemble_probabilities([r["dev_probs"] for r in selected], scout_weights)
    scout_test_probs = ensemble_probabilities([r["test_probs"] for r in selected], scout_weights)

    selection_threshold, scout_val_metrics = find_best_threshold(y_val, scout_val_probs)
    scout_dev_pred = (scout_dev_probs >= selection_threshold).astype(np.int64)
    scout_dev_metrics = evaluate_binary(y_dev, scout_dev_pred)
    scout_ensemble_line = (
        f"[SCOUT ENSEMBLE] val_f1={scout_val_metrics['f1_pos']:.4f} "
        f"dev_f1={scout_dev_metrics['f1_pos']:.4f} thr={selection_threshold:.3f}"
    )
    if y_test is not None:
        scout_test_pred = (scout_test_probs >= selection_threshold).astype(np.int64)
        scout_local_test_metrics = evaluate_binary(y_test, scout_test_pred)
        scout_ensemble_line += f" local_test_f1={scout_local_test_metrics['f1_pos']:.4f}"
    else:
        scout_local_test_metrics = None
    print(scout_ensemble_line)

    final_runs = []
    for run, weight in zip(selected, scout_weights.tolist()):
        run_dir = RUNS_DIR / "final" / sanitize_name(run["model_name"]) / f"seed_{run['seed']}"
        print(f"[FINAL] model={run['model_name']} seed={run['seed']} weight={weight:.4f}")

        _, probs = train_and_predict(
            model_name=run["model_name"],
            seed=run["seed"],
            train_df=train_df,
            eval_df=None,
            predict_dfs={"dev": dev_df, "test": test_df},
            run_dir=run_dir,
            evaluate_during_training=False,
            save_model_artifacts=True,
        )

        dev_pred = (probs["dev"] >= selection_threshold).astype(np.int64)
        dev_metrics = evaluate_binary(y_dev, dev_pred)
        local_test_metrics = None
        if y_test is not None:
            test_pred = (probs["test"] >= selection_threshold).astype(np.int64)
            local_test_metrics = evaluate_binary(y_test, test_pred)

        final_runs.append(
            {
                "model_name": run["model_name"],
                "seed": run["seed"],
                "weight": weight,
                "dev_metrics": dev_metrics,
                "local_test_metrics": local_test_metrics,
                "dev_probs": probs["dev"],
                "test_probs": probs["test"],
                "run_dir": run_dir,
            }
        )

        final_done_line = (
            f"[FINAL DONE] model={run['model_name']} seed={run['seed']} "
            f"dev_f1={dev_metrics['f1_pos']:.4f} thr={selection_threshold:.3f}"
        )
        if local_test_metrics is not None:
            final_done_line += f" local_test_f1={local_test_metrics['f1_pos']:.4f}"
        print(final_done_line)

    final_weights = normalize_weights([r["weight"] for r in final_runs])
    final_dev_probs = ensemble_probabilities([r["dev_probs"] for r in final_runs], final_weights)
    final_test_probs = ensemble_probabilities([r["test_probs"] for r in final_runs], final_weights)

    final_dev_pred_at_selection = (final_dev_probs >= selection_threshold).astype(np.int64)
    final_dev_metrics_at_selection = evaluate_binary(y_dev, final_dev_pred_at_selection)

    final_threshold, final_dev_metrics = find_best_threshold(y_dev, final_dev_probs)
    final_dev_pred = (final_dev_probs >= final_threshold).astype(np.int64)
    final_test_pred = (final_test_probs >= final_threshold).astype(np.int64)
    final_local_test_metrics = None
    if y_test is not None:
        final_local_test_metrics = evaluate_binary(y_test, final_test_pred)

    final_line = (
        f"[FINAL ENSEMBLE] selection_thr={selection_threshold:.3f} "
        f"dev_f1@selection={final_dev_metrics_at_selection['f1_pos']:.4f} "
        f"final_thr={final_threshold:.3f} dev_f1@final={final_dev_metrics['f1_pos']:.4f}"
    )
    if final_local_test_metrics is not None:
        final_line += f" local_test_f1={final_local_test_metrics['f1_pos']:.4f}"
    print(final_line)

    dev_path = OUTPUT_DIR / "dev.txt"
    test_path = OUTPUT_DIR / "test.txt"
    write_prediction_file(final_dev_pred, dev_path)
    write_prediction_file(final_test_pred, test_path)

    if len(final_dev_pred) != EXPECTED_DEV_LINES:
        raise RuntimeError(f"dev.txt has {len(final_dev_pred)} lines, expected {EXPECTED_DEV_LINES}")
    if len(final_test_pred) != EXPECTED_TEST_LINES:
        raise RuntimeError(f"test.txt has {len(final_test_pred)} lines, expected {EXPECTED_TEST_LINES}")

    moved_models = move_selected_models_to_bestmodel(final_runs)

    summary = {
        "config": {
            "model_names": MODEL_NAMES,
            "seeds": SEEDS,
            "top_k": len(selected),
            "ensemble_weighting": ENSEMBLE_WEIGHTING,
            "max_length": MAX_LENGTH,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "num_train_epochs": NUM_TRAIN_EPOCHS,
            "warmup_ratio": WARMUP_RATIO,
            "weighted_sampler": USE_WEIGHTED_SAMPLER,
            "scheduler": "cosine_annealing",
        },
        "internal_split": {
            "folds": INTERNAL_FOLDS,
            "split_seed": SPLIT_SEED,
            "train_inner_size": int(len(inner_train_df)),
            "val_inner_size": int(len(inner_val_df)),
        },
        "selection_threshold": float(selection_threshold),
        "final_threshold": float(final_threshold),
        "scout_selected_runs": [
            {
                "model_name": r["model_name"],
                "seed": int(r["seed"]),
                "val_metrics": r["val_metrics"],
                "dev_metrics": r["dev_metrics"],
                "tuned_threshold_single_run": float(r["tuned_threshold"]),
            }
            for r in selected
        ],
        "scout_ensemble_metrics": {
            "val": scout_val_metrics,
            "dev": scout_dev_metrics,
        },
        "final_seed_runs": [
            {
                "model_name": r["model_name"],
                "seed": int(r["seed"]),
                "weight": float(r["weight"]),
                "dev_metrics": r["dev_metrics"],
            }
            for r in final_runs
        ],
        "final_ensemble_metrics": {
            "dev_at_selection_threshold": final_dev_metrics_at_selection,
            "dev": final_dev_metrics,
        },
        "output_files": {
            "dev_txt": str(dev_path),
            "test_txt": str(test_path),
            "dev_line_count": int(len(final_dev_pred)),
            "test_line_count": int(len(final_test_pred)),
        },
        "moved_model_weights": moved_models,
    }
    if final_local_test_metrics is not None:
        summary["scout_ensemble_metrics"]["local_test"] = scout_local_test_metrics
        for run in summary["scout_selected_runs"]:
            for original in selected:
                if run["model_name"] == original["model_name"] and run["seed"] == int(original["seed"]):
                    run["local_test_metrics"] = original["local_test_metrics"]
                    break
        for run in summary["final_seed_runs"]:
            for original in final_runs:
                if run["model_name"] == original["model_name"] and run["seed"] == int(original["seed"]):
                    run["local_test_metrics"] = original["local_test_metrics"]
                    break
        summary["final_ensemble_metrics"]["local_test"] = final_local_test_metrics

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 80)
    print("Finished.")
    final_report = {
        "selection_threshold": float(selection_threshold),
        "final_threshold": float(final_threshold),
        "final_dev_f1_pos": float(final_dev_metrics["f1_pos"]),
        "dev_txt": str(dev_path),
        "test_txt": str(test_path),
        "weights_dir": str(MODEL_WEIGHTS_DIR),
    }
    if final_local_test_metrics is not None:
        final_report["final_local_test_f1_pos"] = float(final_local_test_metrics["f1_pos"])
    print(json.dumps(final_report, indent=2))


main()
