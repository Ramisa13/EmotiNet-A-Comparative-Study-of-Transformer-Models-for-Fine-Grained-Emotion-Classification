import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from sklearn.metrics import f1_score, accuracy_score
import csv, os, json


# ===== 数据加载 =====
def load_and_merge(path):
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str).str.strip()
    df["emotion"] = df["emotion"].astype(str).str.strip()
    df = df.groupby("text")["emotion"].apply(lambda x: list(set(x))).reset_index()
    return df


train_df = load_and_merge("./train.csv")
val_df = load_and_merge("./validation.csv")
test_df = load_and_merge("./test.csv")

print(f"Loaded data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

# ===== 标签体系 =====
all_labels = sorted({e for sub in train_df["emotion"] for e in sub})
label2id = {l: i for i, l in enumerate(all_labels)}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(label2id)


def encode_labels(emotions):
    v = np.zeros(num_labels)
    for e in emotions:
        if e in label2id:
            v[label2id[e]] = 1
    return v


for df in [train_df, val_df, test_df]:
    df["labels"] = df["emotion"].apply(encode_labels)

# ===== Dataset 定义 =====
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


class GoEmotionDataset(Dataset):
    def __init__(self, df):
        self.texts = df["text"].tolist()
        self.labels = torch.tensor(np.stack(df["labels"].to_numpy()), dtype=torch.float)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": self.labels[idx],
        }


train_ds = GoEmotionDataset(train_df)
val_ds = GoEmotionDataset(val_df)
test_ds = GoEmotionDataset(test_df)

# ===== 模型加载 =====
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=num_labels,
    problem_type="multi_label_classification"
)

# ===== 评估指标 =====
def jaccard_score_example_based(y_true, y_pred):
    inter = (y_true & y_pred).sum(axis=1)
    union = (y_true | y_pred).sum(axis=1)
    union = np.where(union == 0, 1, union)
    return (inter / union).mean()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    y_pred = (probs > 0.5).astype(int)
    y_true = labels.astype(int)
    return {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "subset_acc": accuracy_score(y_true, y_pred),
        "jaccard": jaccard_score_example_based(y_true, y_pred),
    }


# ===== 日志保存 Callback =====
class CSVLoggerCallback(TrainerCallback):
    def __init__(self, path="epoch_metrics.csv"):
        self.path = path
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["epoch", "eval_loss", "micro_f1", "macro_f1", "jaccard", "subset_acc"]
                )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                state.epoch,
                metrics.get("eval_loss"),
                metrics.get("eval_micro_f1"),
                metrics.get("eval_macro_f1"),
                metrics.get("eval_jaccard"),
                metrics.get("eval_subset_acc"),
            ])


# ===== 训练参数 =====
args = TrainingArguments(
    output_dir="./roberta-goemotions",
    num_train_epochs=20,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="micro_f1",
    greater_is_better=True,
    logging_steps=50,
    report_to=[],
    save_total_limit=3
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[CSVLoggerCallback("epoch_metrics.csv")]
)

print(f"Training on GPU: {torch.cuda.get_device_name(0)}")

# ===== 开始训练 =====
trainer.train()
trainer.save_model("./roberta-goemotions-final")
tokenizer.save_pretrained("./roberta-goemotions-final")

# ===== 测试集评估 =====
print("Training complete. Evaluating on test set...")
res = trainer.predict(test_ds)
print(res.metrics)

with open("test_metrics.json", "w") as f:
    json.dump(res.metrics, f, indent=4)
print("Test metrics saved to test_metrics.json")
