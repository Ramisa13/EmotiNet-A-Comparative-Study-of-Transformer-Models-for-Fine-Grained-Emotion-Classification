import os
import random
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset, ClassLabel
import evaluate


# ---------------------------
# Config / Hyperparameters
# ---------------------------
MODEL_NAME = "FacebookAI/xlm-roberta-base"   # model to fine-tune
OUTPUT_DIR = "xlm_roberta_finetuned"
SEED = 42

# CSV file paths (edit or pass via args)
TRAIN_CSV = "/scratch/project_2011211/Fahim/affective_computing/train.csv"
VAL_CSV = "/scratch/project_2011211/Fahim/affective_computing/validation.csv"
TEST_CSV = "/scratch/project_2011211/Fahim/affective_computing/test.csv"

# Column names in CSVs
TEXT_COL = "text"
LABEL_COL = "emotion"

# Training hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_LENGTH = 512
WARMUP_STEPS = 0
LOGGING_STEPS = 100
SAVE_STRATEGY = "epoch"  # or "steps"
METRIC_FOR_BEST_MODEL = "eval_loss"  # or "f1"
LOAD_BEST_AT_END = True

# ---------------------------
# Utility and Setup
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------------------
# Load data
# ---------------------------
def load_csv_to_hf_dataset(path: str, text_col=TEXT_COL, label_col=LABEL_COL):
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV at {path} must contain columns: {text_col}, {label_col}")
    # Drop NaNs in required columns
    df = df[[text_col, label_col]].dropna()
    return df

train_df = load_csv_to_hf_dataset(TRAIN_CSV)
val_df   = load_csv_to_hf_dataset(VAL_CSV)
test_df  = load_csv_to_hf_dataset(TEST_CSV)

print("Train size:", len(train_df))
print("Val size:", len(val_df))
print("Test size:", len(test_df))

# ---------------------------
# Label encoding
# ---------------------------
label_encoder = LabelEncoder()
# Fit on all labels to ensure consistent mapping
all_labels = pd.concat([train_df[LABEL_COL], val_df[LABEL_COL], test_df[LABEL_COL]])
label_encoder.fit(all_labels)

def encode_labels(df: pd.DataFrame):
    df = df.copy()
    df["label"] = label_encoder.transform(df[LABEL_COL])
    return df

train_df = encode_labels(train_df)
val_df   = encode_labels(val_df)
test_df  = encode_labels(test_df)

num_labels = len(label_encoder.classes_)
print("Labels (index -> label):")
for i, lbl in enumerate(label_encoder.classes_):
    print(i, "->", lbl)
print("Num labels:", num_labels)

# ---------------------------
# Create Hugging Face Datasets
# ---------------------------
hf_train = Dataset.from_pandas(train_df[[TEXT_COL, "label"]].rename(columns={TEXT_COL: "text"}))
hf_val   = Dataset.from_pandas(val_df[[TEXT_COL, "label"]].rename(columns={TEXT_COL: "text"}))
hf_test  = Dataset.from_pandas(test_df[[TEXT_COL, "label"]].rename(columns={TEXT_COL: "text"}))

# Remove the pandas index column if present
for d in (hf_train, hf_val, hf_test):
    if "__index_level_0__" in d.column_names:
        d = d.remove_columns("__index_level_0__")

# ---------------------------
# Tokenizer & Tokenization
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

hf_train = hf_train.map(tokenize_fn, batched=True)
hf_val   = hf_val.map(tokenize_fn, batched=True)
hf_test  = hf_test.map(tokenize_fn, batched=True)

# Set format for PyTorch
columns = ["input_ids", "attention_mask", "label"]
hf_train.set_format(type="torch", columns=columns)
hf_val.set_format(type="torch", columns=columns)
hf_test.set_format(type="torch", columns=columns)

# ---------------------------
# Model
# ---------------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

# If GPU present, move model to GPU
model.to(device)

# ---------------------------
# Metrics function for Trainer (basic)
# ---------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# ---------------------------
# TrainingArguments & Trainer
# ---------------------------
# Compute steps per epoch for evaluation and checkpoint saving
steps_per_epoch = max(len(hf_train) // BATCH_SIZE, 1)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=WARMUP_STEPS,
    logging_steps=LOGGING_STEPS,
    
    # Old versions: use eval_steps instead of evaluation_strategy
    eval_steps=steps_per_epoch,        # evaluate roughly once per epoch

    # Old versions: use save_steps instead of save_strategy
    save_steps=steps_per_epoch,        # save checkpoint roughly once per epoch
    save_total_limit=2,                # keep only last 2 checkpoints

    # Optional: disable wandb / other loggers
    report_to=[],                       # prevents wandb from being used

    # Old versions may not support these, so remove:
    # load_best_model_at_end, metric_for_best_model, greater_is_better, push_to_hub

    seed=SEED,
    fp16=torch.cuda.is_available(),    # enable mixed precision if GPU available
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_train,
    eval_dataset=hf_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ---------------------------
# Train
# ---------------------------
print("Starting training ...")
trainer.train()
print("Training finished.")

# Save model + tokenizer + label mapping
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
# Save label encoder classes for inference
label_map_path = os.path.join(OUTPUT_DIR, "label_mapping.txt")
with open(label_map_path, "w", encoding="utf-8") as f:
    for i, lbl in enumerate(label_encoder.classes_):
        f.write(f"{i}\t{lbl}\n")
print(f"Saved label mapping to {label_map_path}")

# ---------------------------
# Evaluate on train / val / test and print classification reports
# ---------------------------
def predict_and_report(dataset_hf, split_name: str):
    print(f"\n=== Evaluating on {split_name} ===")
    # Use trainer.predict (it returns PredictionsTuple with predictions, label_ids, metrics)
    predictions_result = trainer.predict(dataset_hf)
    logits = predictions_result.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    labels = predictions_result.label_ids

    # Classification report (sklearn)
    target_names = list(label_encoder.classes_)
    report = classification_report(labels, preds, target_names=target_names, zero_division=0)
    print(f"\nClassification report for {split_name}:\n")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(f"Confusion Matrix - {split_name}")
    outpath = os.path.join(OUTPUT_DIR, f"confusion_matrix_{split_name}.png")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"Saved confusion matrix image to {outpath}")

    # Also print overall accuracy & macro/f1
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    print(f"{split_name} Accuracy: {acc:.4f} | Weighted Precision: {prec:.4f} | Weighted Recall: {rec:.4f} | Weighted F1: {f1:.4f}")

# Predict & report for train, val, test
predict_and_report(hf_train, "train")
predict_and_report(hf_val, "validation")
predict_and_report(hf_test, "test")

print("\nAll done. Model + tokenizer + artifacts saved to:", OUTPUT_DIR)
print("Label index -> label mapping in:", label_map_path)
