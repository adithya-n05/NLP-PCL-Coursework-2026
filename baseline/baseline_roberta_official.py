#!/usr/bin/env python3

import ast
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# -----------------------------------------------------------------------------
# Changed constants
# -----------------------------------------------------------------------------

MAIN_FILE = Path("data/dontpatronizeme_pcl.tsv")
TRAIN_SPLIT_FILE = Path("data/practice splits/train_semeval_parids-labels.csv")
DEV_SPLIT_FILE = Path("data/practice splits/dev_semeval_parids-labels.csv")

MODEL_NAME = "roberta-base"
SEED = 42

# -----------------------------------------------------------------------------
# Fixed constants
# -----------------------------------------------------------------------------

NEG_TO_POS_RATIO = 2
BASELINE_MAX_LENGTH = 128
BASELINE_EPOCHS = 1.0
BASELINE_TRAIN_BATCH_SIZE = 8
BASELINE_EVAL_BATCH_SIZE = 100
BASELINE_LR = 4e-5
BASELINE_WARMUP_RATIO = 0.06

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.dont_write_bytecode = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_binary(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_pos": float(p),
        "recall_pos": float(r),
        "f1_pos": float(f1),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def to_hf_dataset(df, label_col=None):
    data = {"text": df["text"].astype(str).tolist()}
    if label_col is not None:
        data["labels"] = df[label_col].astype(int).tolist()
    return Dataset.from_dict(data)


def load_data():
    main_df = pd.read_csv(
        MAIN_FILE,
        sep="\t",
        header=None,
        names=["par_id", "article_id", "keyword", "country", "text", "label_0_4"],
    )
    main_df["par_id"] = pd.to_numeric(main_df["par_id"], errors="coerce").astype(int)
    main_df["text"] = main_df["text"].fillna("")

    train_split = pd.read_csv(TRAIN_SPLIT_FILE)
    dev_split = pd.read_csv(DEV_SPLIT_FILE)

    for split_df in (train_split, dev_split):
        split_df["par_id"] = pd.to_numeric(split_df["par_id"], errors="coerce").astype(int)
        split_df["label_bin"] = split_df["label"].apply(ast.literal_eval).apply(lambda x: int(any(x)))

    train_full = train_split[["par_id", "label_bin"]].merge(
        main_df[["par_id", "text"]], on="par_id", how="left", sort=False, validate="1:1"
    )
    dev_df = dev_split[["par_id", "label_bin"]].merge(
        main_df[["par_id", "text"]], on="par_id", how="left", sort=False, validate="1:1"
    )

    if train_full["text"].isna().any() or dev_df["text"].isna().any():
        raise ValueError("Missing rows while rebuilding official splits from par_id.")

    pos = train_full[train_full["label_bin"] == 1]
    neg = train_full[train_full["label_bin"] == 0].iloc[: len(pos) * NEG_TO_POS_RATIO]
    train_downsampled = pd.concat([pos, neg], axis=0).reset_index(drop=True)

    return train_full, train_downsampled, dev_df


def main():
    set_seed(SEED)
    run_dir = Path("baseline/.tmp_roberta_official_check")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_full, train_downsampled, dev_df = load_data()

    train_ds = to_hf_dataset(train_downsampled, label_col="label_bin")
    dev_ds = to_hf_dataset(dev_df, label_col="label_bin")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=BASELINE_MAX_LENGTH)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    dev_ds = dev_ds.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=BASELINE_EPOCHS,
        per_device_train_batch_size=BASELINE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=BASELINE_EVAL_BATCH_SIZE,
        learning_rate=BASELINE_LR,
        warmup_ratio=BASELINE_WARMUP_RATIO,
        weight_decay=0.0,
        evaluation_strategy="no",
        save_strategy="no",
        logging_strategy="no",
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        fp16=torch.cuda.is_available(),
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        tokenizer=tokenizer,
    )

    trainer.train()
    dev_logits = trainer.predict(dev_ds).predictions
    dev_preds = np.argmax(dev_logits, axis=-1)
    dev_metrics = evaluate_binary(dev_df["label_bin"].to_numpy(dtype=np.int64), dev_preds)

    print("Baseline sanity check")
    print(f"train_full={len(train_full)} train_downsampled={len(train_downsampled)} dev={len(dev_df)}")
    print(
        f"dev_f1_pos={dev_metrics['f1_pos']:.4f} "
        f"precision_pos={dev_metrics['precision_pos']:.4f} "
        f"recall_pos={dev_metrics['recall_pos']:.4f}"
    )


main()
