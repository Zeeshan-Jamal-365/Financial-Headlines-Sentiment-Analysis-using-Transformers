# Comparative Analysis of Transformer Models for Financial Headline Sentiment Classification

## Overview
This project fine-tunes and compares five transformer models for **3-class sentiment classification** on **financial news headlines**: **BERT, DistilBERT, ALBERT, RoBERTa, and FinBERT**. The objective is to evaluate the models under a consistent training setup and a robust validation strategy.

**Primary metric:** Macro-averaged F1 (Macro-F1), used because it gives each class equal weight and is more reliable under class imbalance.

---

## Models Evaluated
- BERT
- DistilBERT
- ALBERT
- RoBERTa
- FinBERT

---

## Method

### Cross-validation
- **Stratified 5-fold cross-validation** is used to keep the class distribution consistent across folds and reduce dependence on a single split.
- For each fold, the **best checkpoint** is selected using **validation Macro-F1**, then evaluated and averaged across folds.

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
- Compute: Kaggle notebook, NVIDIA T4 ×2  

---

## Metrics
Reported metrics:
- Accuracy
- Precision
- Recall
- Macro-F1 (primary)

---

## Results (mean across 5 folds)
> Note: In the report table, BERT is shown as a percentage while other models are shown as decimals.  
> Below are the values as reported and also as readable percentages for consistency.

### As reported
- **BERT:** Accuracy 88.41%, Precision 88.40%, Recall 88.41%, F1 88.34%
- **DistilBERT:** Accuracy 0.8797, Precision 0.8806, Recall 0.8797, F1 0.8791
- **ALBERT:** Accuracy 0.8808, Precision 0.8820, Recall 0.8808, F1 0.8806
- **RoBERTa:** Accuracy 0.8934, Precision 0.8935, Recall 0.8934, F1 0.8927
- **FinBERT:** Accuracy 0.9025, Precision 0.9030, Recall 0.9025, F1 0.9021

### Readable percent equivalents
- **DistilBERT:** Accuracy 87.97%, Precision 88.06%, Recall 87.97%, Macro-F1 87.91%
- **ALBERT:** Accuracy 88.08%, Precision 88.20%, Recall 88.08%, Macro-F1 88.06%
- **RoBERTa:** Accuracy 89.34%, Precision 89.35%, Recall 89.34%, Macro-F1 89.27%
- **FinBERT:** Accuracy 90.25%, Precision 90.30%, Recall 90.25%, Macro-F1 90.21%

---

## Statistical Testing (fold-level Macro-F1)
Performance differences are tested using fold-level Macro-F1:
- Paired t-tests
- One-way ANOVA (F = 5.345, p = 0.00429)
- Tukey HSD post-hoc analysis

Examples reported:
- Paired t-test: BERT vs FinBERT is significant (p = 0.00674)
- Tukey HSD:
