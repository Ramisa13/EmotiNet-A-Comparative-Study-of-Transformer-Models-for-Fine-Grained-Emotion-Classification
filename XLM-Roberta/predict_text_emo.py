import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ===== 1. 路径 =====
CHECKPOINT = "output/checkpoint-8920"          # 最佳模型目录
LABEL_DIR  = "dataset"                         # 存放 label2id.json / id2label.json 的地方

# ===== 2. 读标签映射 =====
with open(os.path.join(LABEL_DIR, "label2id.json")) as f:
    label2id = json.load(f)
with open(os.path.join(LABEL_DIR, "id2label.json")) as f:
    id2label = json.load(f)
id2label = {int(k): v for k, v in id2label.items()}
num_labels = len(id2label)

# ===== 3. 加载分词器 + 模型 =====
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT,
    num_labels=num_labels,
    problem_type="multi_label_classification"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ===== 4. 推理函数 =====
@torch.no_grad()
def predict(text: str, threshold: float = 0.5):
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    ).to(device)

    logits = model(**enc).logits          # [1, num_labels]
    probs  = torch.sigmoid(logits[0])     # [num_labels]
    preds  = (probs > threshold).cpu().numpy()

    emotions = [id2label[i] for i, flag in enumerate(preds) if flag]
    return emotions, probs.cpu().numpy()

# ===== 5. 单句 demo =====
if __name__ == "__main__":
    while True:
        txt = input("\n输入句子 (q 退出): ").strip()
        if txt.lower() == "q":
            break
        emo, score = predict(txt, 0.1)
        print("预测情绪:", emo)