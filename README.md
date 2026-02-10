# Comparative Analysis of Transformer Models for Financial Headline Sentiment Classification

## Overview
This project fine-tunes and compares five transformer models for **3-class sentiment classification** on **financial news headlines**: **BERT, DistilBERT, ALBERT, RoBERTa, and FinBERT**. The objective is to evaluate the models under a consistent training setup and a robust validation strategy. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

**Primary metric:** Macro-averaged F1 (Macro-F1), used because it gives each class equal weight and is more reliable under class imbalance. :contentReference[oaicite:2]{index=2}

---

## Models Evaluated
- BERT  
- DistilBERT  
- ALBERT  
- RoBERTa  
- FinBERT :contentReference[oaicite:3]{index=3}

---

## Method

### Cross-validation
- **Stratified 5-fold cross-validation** is used to keep the class distribution consistent across folds and reduce dependence on a single split. :contentReference[oaicite:4]{index=4}  
- For each fold, the **best checkpoint is selected by validation Macro-F1**, then evaluated and averaged across folds. :contentReference[oaicite:5]{index=5}

### Unified training configuration (applied to all models)
- Framework: Hugging Face Transformers  
- Optimizer: AdamW  
- Learning rate: 3 × 10⁻⁵  
- Epochs: 10 (max)  
- Batch size: 32 per GPU (effective 64 with 2 GPUs)  
- Weight decay: 0.01  
- Warmup ratio: 0.1  
- FP16: enabled  
- Early stopping: enabled  
- Compute: Kaggle notebook, NVIDIA T4 ×2 :contentReference[oaicite:6]{index=6}

---

## Metrics
Reported metrics:
- Accuracy  
- Precision  
- Recall  
- Macro-F1 (primary) :contentReference[oaicite:7]{index=7}

---

## Results (mean across 5 folds)
Table 4.1 in the report lists mean cross-validation metrics. :contentReference[oaicite:8]{index=8}

> Note: In Table 4.1, BERT is displayed as a percentage, while other models are shown as decimals with a “%” sign.  
> Below, values are shown **as reported** and also as **readable percent equivalents**.

### As reported (Table 4.1)
- BERT: Accuracy 88.41%, Precision 88.40%, Recall 88.41%, F1 88.34%  
- DistilBERT: Accuracy 0.8797, Precision 0.8806, Recall 0.8797, F1 0.8791  
- ALBERT: Accuracy 0.8808, Precision 0.8820, Recall 0.8808, F1 0.8806  
- RoBERTa: Accuracy 0.8934, Precision 0.8935, Recall 0.8934, F1 0.8927  
- FinBERT: Accuracy 0.9025, Precision 0.9030, Recall 0.9025, F1 0.9021 :contentReference[oaicite:9]{index=9}

### Readable percent equivalents
- DistilBERT: Accuracy 87.97%, Precision 88.06%, Recall 87.97%, Macro-F1 87.91%  
- ALBERT: Accuracy 88.08%, Precision 88.20%, Recall 88.08%, Macro-F1 88.06%  
- RoBERTa: Accuracy 89.34%, Precision 89.35%, Recall 89.34%, Macro-F1 89.27%  
- FinBERT: Accuracy 90.25%, Precision 90.30%, Recall 90.25%, Macro-F1 90.21% :contentReference[oaicite:10]{index=10}

---

## Statistical Testing (fold-level Macro-F1)
The report tests whether performance differences are statistically meaningful using fold-level Macro-F1:
- Paired t-tests (Table 5.1) :contentReference[oaicite:11]{index=11}  
- One-way ANOVA across all five models (F = 5.345, p = 0.00429) :contentReference[oaicite:12]{index=12}  
- Tukey HSD post-hoc analysis (Table 5.2) :contentReference[oaicite:13]{index=13}  

Examples:
- Paired t-test: BERT vs FinBERT is significant (p = 0.00674). :contentReference[oaicite:14]{index=14}  
- Tukey HSD: FinBERT vs DistilBERT (adjusted p = 0.0074), FinBERT vs ALBERT (adjusted p = 0.0130), FinBERT vs BERT (adjusted p = 0.0352). :contentReference[oaicite:15]{index=15}  

---

## Suggested Repository Structure
- `notebooks/`  
  - training and evaluation notebooks
- `src/`  
  - preprocessing, training, and evaluation utilities
- `results/`  
  - tables, plots, confusion matrices, ROC-AUC outputs :contentReference[oaicite:16]{index=16}

---

## How to Run (template)
1. Install dependencies: PyTorch, Transformers, scikit-learn, pandas.  
2. Run training using the unified configuration.  
3. Evaluate with stratified 5-fold CV.  
4. Aggregate fold metrics and export results. :contentReference[oaicite:17]{index=17}

---

## Reference
This repository is based on the project report: **"Comparative Analysis of Transformer Models for Financial Headline Sentiment Classification."** :contentReference[oaicite:18]{index=18}
