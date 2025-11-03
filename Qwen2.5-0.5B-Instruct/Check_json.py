import json

def count_multilabel(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    multi = 0
    for item in data:
        labels = item["output"].strip()
        if "," in labels:   # 有逗号代表多标签
            multi += 1

    print(f" {json_path}")
    print(f"   Total samples: {total}")
    print(f"   Multi-label samples: {multi} ({multi/total:.2%})")
    print("-" * 60)
    return total, multi

if __name__ == "__main__":
    total_all = multi_all = 0
    for path in ["emotion_train.json", "emotion_validation.json", "emotion_test.json"]:
        t, m = count_multilabel(path)
        total_all += t
        multi_all += m
    print(f" Overall total: {total_all} | Multi-label: {multi_all} ({multi_all/total_all:.2%})")
