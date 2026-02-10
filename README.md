# Financial Headline Sentiment Analysis (Transformer Model Comparison)

This repository trains and compares transformer models to classify **financial news headlines** into **Negative / Neutral / Positive** sentiment.  
Models compared: **BERT, DistilBERT, ALBERT, RoBERTa, FinBERT**.

**Best model (reported): FinBERT** — **Accuracy 90.25%**, **Macro-F1 90.21%** (mean across stratified 5-fold CV).

---

## What this project shows (skills)
- **Data preparation & dataset versioning:** baseline dataset (`news.csv`) and augmented dataset (`news_augmented.csv`)
- **Model training & evaluation:** stratified **5-fold cross-validation** with **Macro-F1** as the primary metric
- **Results reporting:** comparison table + statistical validation (t-tests, ANOVA, Tukey HSD)
- **Practical inference:** notebook demo to score new headlines with FinBERT

---

## Repository contents

### Data
- `news.csv` — main dataset used for training/evaluation
- `news_augmented.csv` — augmented dataset used to reduce class imbalance and improve stability

### Notebooks
Training (one notebook per model):
- `financial-headline-sentiment-analysisbert3.ipynb` — BERT
- `financial-headline-sentiment-analysis-distilbert3.ipynb` — DistilBERT
- `financial-headline-sentiment-analysis-albert3.ipynb` — ALBERT
- `financial-headline-sentiment-analysisroberta.ipynb` — RoBERTa
- `financial-headline-sentiment-analysis-finbert.ipynb` — FinBERT

Evaluation / visuals:
- `model-comparison-2.ipynb` — aggregates metrics and compares models
- `financial-headline-sentiment-analysis-viz.ipynb` — plots and visualizations

Inference:
- `finbert-deployment-inference.ipynb` — inference demo (score new headlines)

---

## Method

### Cross-validation
- Uses **stratified 5-fold cross-validation** to keep class distribution consistent across folds.
- For each fold, selects the **best checkpoint** using **validation Macro-F1**, then averages results across folds.

### Metric choice
- **Macro-F1** is the primary metric because it treats each class equally and is more informative than accuracy when data is imbalanced.

---

## Results (mean across 5 folds)
> Note: some report values were written as decimals (e.g., 0.9021), which correspond to percentages (90.21%).

| Model | Accuracy | Precision | Recall | Macro-F1 |
|---|---:|---:|---:|---:|
| BERT | 88.41% | 88.40% | 88.41% | 88.34% |
| DistilBERT | 87.97% | 88.06% | 87.97% | 87.91% |
| ALBERT | 88.08% | 88.20% | 88.08% | 88.06% |
| RoBERTa | 89.34% | 89.35% | 89.34% | 89.27% |
| **FinBERT** | **90.25%** | **90.30%** | **90.25%** | **90.21%** |

---

## Statistical validation
To confirm differences are not due to a single random split, the report includes:
- **paired t-tests**
- **one-way ANOVA** (F = 5.345, p = 0.00429)
- **Tukey HSD** post-hoc analysis

---

## Quick start
1. Open **`model-comparison-2.ipynb`** to see aggregated comparison results.
2. Open **`financial-headline-sentiment-analysis-viz.ipynb`** for plots.
3. Open **`financial-headline-sentiment-analysis-finbert.ipynb`** to view the best model training run.
4. Open **`finbert-deployment-inference.ipynb`** to score new headlines.

---

## Example inference
Use `finbert-deployment-inference.ipynb` to score new text headlines.

Example format:
- Input: `Company reports record profits as demand surges`
- Output: `Positive` (with a confidence score)

---

## Running locally
These notebooks were originally run on Kaggle GPU. For local runs you may need:

- Python 3.9+
- `torch`
- `transformers`
- `datasets`
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`

---

## Related reference (external)
https://github.com/prsvnkt/Sentiment-Analysis-Financial-News

---

## License
See `LICENSE`.

