
# EmotiNet: A Comparative Study of Transformer Models for Fine-Grained Emotion Classification

This repository contains the code and results for **EmotiNet**, a comprehensive comparative study on fine-tuning state-of-the-art Transformer models for multi-label emotion classification. The study focuses on the highly imbalanced **GoEmotions** dataset, which includes 58,000 Reddit comments labeled with 27 distinct emotions and a neutral category.

The goal of this project is to investigate the performance, efficiency, and architectural impact of various popular and novel BERT-family models, alongside a decoder-only Large Language Model (LLM), in detecting emotions from text.

## 🎯 Key Findings & Best Performers

The study evaluated five encoder-only Transformer models (BERT, DistilBERT, RoBERTa, XLM-RoBERTa, ModernBERT) and one decoder-only LLM (**Qwen2.5-0.5B-Instruct**) for emotion classification.

### **Performance Highlights**:
- **XLM-RoBERTa** (Multi-Label Classification):
    - **Weighted F1**: 0.551
    - Strongest overall performance in the multi-label task.
    
- **ModernBERT** (Single-Label Classification):
    - **Weighted F1**: 0.627
    - Best performing BERT variant due to architectural improvements like **RoPE** (Rotary Positional Embeddings) and **GEGLU** activation.

- **DistilBERT** (Single-Label Classification):
    - **Accuracy**: 0.598
    - Best choice for production and real-time inference due to its efficiency and small size.

- **Data Balancing** (Multi-Label Classification):
    - **Macro F1 Improvement**: +25.5%
    - Demonstrates the critical need to mitigate class imbalance to improve detection of rare emotions.


## 🚀 Getting Started

### 1. Prerequisites

* Python 3.9+
* NVIDIA GPU (recommended for efficient training)

### 2. Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/Ramisa13/EmotiNet-A-Comparative-Study-of-Transformer-Models-for-Fine-Grained-Emotion-Classification.git
cd EmotiNet-A-Comparative-Study-of-Transformer-Models-for-Fine-Grained-Emotion-Classification
pip install -r requirements.txt
```

### 3. Data Preparation

Ensure that the **GoEmotions** dataset (`goemotions_dataset.csv`) is present in the `data/` directory.

### 4. Training a Model

You can fine-tune any of the models using the main training script. For example, to train **XLM-RoBERTa** for the multi-label task:

```bash
python src/train.py \
    --model_name "xlm-roberta-base" \
    --task_type "multi-label" \
    --output_dir "./results/xlm_roberta_multilabel" \
    --num_epochs 5
```

For the **DistilBERT** model on the single-label task:

```bash
python src/train.py \
    --model_name "distilbert-base-uncased" \
    --task_type "single-label" \
    --output_dir "./results/distilbert_singlelabel" \
    --num_epochs 4
```

### 5. Evaluating Results

After training, use the notebooks to analyze the performance, visualize confusion matrices, and generate comparative plots.

Open the **evaluation and results notebook** to:

* Load saved model checkpoints.
* Run the full suite of evaluation metrics (e.g., Weighted F1, Macro F1).
* Observe the impact of thresholding on multi-label results.

```bash
open notebooks/3_evaluation_and_results.ipynb
```

## 🤝 Contribution

Feel free to open issues or submit pull requests for:

* New model comparisons
* Improved architectures
* Advanced data balancing techniques (e.g., synthetic data generation for rare classes)

## 📄 Citation

If you use this work in your research, please cite the associated report:

```bibtex
@article{EmotiNet2025,
  title={EmotiNet: A Comparative Study of Transformer Models for Fine-Grained Emotion Classification},
  author={Istiak, Fahim and Luo, Yunxiang and Tapaninaho, Joonas and Zhang, Yuanjun and Rahman, Mohammad Rakibur and Mridha, Ramisa Fariha},
  journal={Report, University of Oulu},
  year={2025}
}
```


