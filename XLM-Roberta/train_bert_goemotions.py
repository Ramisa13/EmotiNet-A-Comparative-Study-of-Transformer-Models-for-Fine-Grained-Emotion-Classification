import os
import random
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import jaccard_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


# ======================
# 配置部分（可修改）
# ======================
PRETRAINED_MODEL_PATH = "./xlm-roberta-base"
DATA_DIR = "dataset"
OUTPUT_DIR = "output"
BATCH_SIZE = 16
EPOCHS = 20
LR = 2e-5
MAX_LEN = 512
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ======================
# 数据读取与处理
# ======================
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # 合并相同text的多标签
    grouped = df.groupby("text")["emotion"].apply(list).reset_index()
    return grouped

def load_label_maps():
    with open(os.path.join(DATA_DIR, "id2label.json"), "r") as f:
        id2label = json.load(f)
    with open(os.path.join(DATA_DIR, "label2id.json"), "r") as f:
        label2id = json.load(f)
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {k: int(v) for k, v in label2id.items()}
    return id2label, label2id

class GoEmotionsDataset(Dataset):
    def __init__(self, df, tokenizer, label2id, max_len=128):
        self.texts = df["text"].tolist()
        self.labels = df["emotion"].tolist()
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        self.num_labels = len(label2id)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        label_vec = np.zeros(self.num_labels)
        for emo in labels:
            label_vec[self.label2id[emo]] = 1.0

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label_vec, dtype=torch.float),
        }

# ======================
# 训练与评估函数
# ======================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    labels = labels.astype(int)
    score = jaccard_score(labels, preds, average="samples")
    return {"jaccard_acc": score}

def evaluate_model(trainer, datasets, names):
    for name, ds in zip(names, datasets):
        metrics = trainer.evaluate(ds)
        print(f"📊 {name} set Jaccard Acc: {metrics['eval_jaccard_acc']:.4f}")

# ======================
# 主训练逻辑
# ======================
def main():
    print("🚀 Loading data...")
    train_df = load_data(os.path.join(DATA_DIR, "train.csv"))
    val_df = load_data(os.path.join(DATA_DIR, "validation.csv"))
    test_df = load_data(os.path.join(DATA_DIR, "test.csv"))

    id2label, label2id = load_label_maps()
    num_labels = len(label2id)

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)

    train_dataset = GoEmotionsDataset(train_df, tokenizer, label2id, MAX_LEN)
    val_dataset = GoEmotionsDataset(val_df, tokenizer, label2id, MAX_LEN)
    test_dataset = GoEmotionsDataset(test_df, tokenizer, label2id, MAX_LEN)

    print(f"Loaded {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples")

    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_PATH,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

    # 使用 BCEWithLogitsLoss
    model.config.hidden_dropout_prob = 0.1

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,  # 只保留一个模型（即最佳模型）
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="jaccard_acc",
        greater_is_better=True,  # 评估指标越大越好，针对Jaccard
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=500,
        report_to="none",
        fp16=True,
        torch_compile=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("🔥 Start training...")
    trainer.train()

    print("✅ Training done. Evaluating...")
    evaluate_model(trainer, [train_dataset, val_dataset, test_dataset], ["Train", "Validation", "Test"])

    print(f"💾 Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()
