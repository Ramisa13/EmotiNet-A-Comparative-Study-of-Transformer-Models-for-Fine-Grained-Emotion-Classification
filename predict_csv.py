import pandas as pd
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

# ===== 5. 对CSV进行处理并添加情感预测 =====
def process_and_predict(csv_file, output_file, threshold=0.5):
    df = pd.read_csv(csv_file)

    # 1. 把空值填成空字符串，并强制转 str
    df['text'] = df['text'].fillna('').astype(str)

    # 2. 初始化 emotion 列为空字符串
    df['emotion'] = ''

    for idx, row in df.iterrows():
        text = row['text'].strip()
        if not text:                       # 跳过纯空串
            df.at[idx, 'emotion'] = ''
            continue

        emotions, _ = predict(text, threshold)
        df.at[idx, 'emotion'] = ', '.join(emotions)

        if idx % 100 == 0:
            print(f"处理了 {idx} 条评论...")

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ 数据处理完成，结果保存到 '{output_file}'")

# ===== 6. 调用函数进行处理 =====
if __name__ == "__main__":
    # 输入处理后的CSV文件路径和输出文件路径
    input_csv = "processed_data_no_empty_text.csv"  # 你之前处理过的CSV文件
    output_csv = "processed_with_emotion.csv"  # 输出的CSV文件

    # 执行情感预测处理
    process_and_predict(input_csv, output_csv, threshold=0.1)
