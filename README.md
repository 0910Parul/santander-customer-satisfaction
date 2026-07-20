# 🏦 Santander Customer Satisfaction — Predicting At-Risk Customers

**Which bank customers are about to leave — before they say a word?**
A machine-learning project on the [Santander Customer Satisfaction](https://www.kaggle.com/competitions/santander-customer-satisfaction) Kaggle dataset that predicts dissatisfied customers from anonymized behavioral data, framed as a customer-retention business problem. Built for the MSBA program at the University of Minnesota (Carlson School of Management) — our team's solution ranked **#1 in class**.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10-blue" />
  <img src="https://img.shields.io/badge/scikit--learn-RandomForest-orange" />
  <img src="https://img.shields.io/badge/pandas-EDA-150458" />
  <img src="https://img.shields.io/badge/Metric-AUC--ROC-success" />
</p>

---

## 📌 Business Problem

Dissatisfied customers rarely complain — they quietly disengage and leave. By the time a bank notices, the relationship is already gone. The goal: **flag at-risk customers early** so the bank can intervene while it still can.

**Why it matters (illustrative economics):** retaining a customer costs 5–7× less than acquiring one. For a large retail bank, identifying even a small share of at-risk customers and saving ~10% of them can protect **hundreds of millions in annual revenue**.

---

## 🎯 Approach

**The key insight — accuracy is the wrong metric.** 96% of customers are satisfied, so a model that predicts "satisfied" for everyone scores 96% accuracy while catching **zero** at-risk customers. We optimized for **AUC-ROC**, which measures how well the model separates the two groups regardless of the 24:1 class imbalance.

My focus on the project was the **Random Forest model, feature engineering, and documentation**:

- **Data cleaning** — removed constant and duplicate features from the 369 anonymized columns
- **Feature engineering** — rule-based flags and one-hot encoding (expanded to ~116 features)
- **EDA** — built a feature-prefix taxonomy and confirmed the `var15` (age) "golden rule" signal
- **Modeling** — tuned a Random Forest classifier (cross-validated AUC ≈ **0.839**)

---

## 📊 Dataset

| Metric | Value |
|---|---|
| Training rows | 76,020 |
| Test rows | 75,818 |
| Raw features | 369 anonymized numeric |
| Dissatisfied (TARGET=1) | 3.9% (~24:1 imbalance) |
| Evaluation metric | AUC-ROC |

Data is public via Kaggle and **not stored in this repo** — download `train.csv` / `test.csv` from the [competition page](https://www.kaggle.com/competitions/santander-customer-satisfaction/data).

---

## 🔑 Results

- **Final private leaderboard AUC: 0.82745** — ranked **#1 in class**
- Demonstrated that disciplined cleaning + feature engineering on anonymized data, evaluated with the right metric, beats naive high-accuracy baselines.

---

## 🗂️ Repository Structure

```
santander-customer-satisfaction/
├── README.md
├── Top100_Features_correlation.ipynb   # feature analysis & correlation study
├── data_dictionary.md                  # feature groups / prefix taxonomy
├── rf_submission_final.csv             # Random Forest submission
├── requirements.txt
└── .gitignore
```

---

## 🛠️ Skills Demonstrated

`Data Cleaning` · `Feature Engineering` · `Class Imbalance` · `Random Forest` · `Model Evaluation (AUC-ROC)` · `EDA` · `Business Framing`

---

## 👤 About

Team competition project (MSBA, Carlson School of Management). My contribution: Random Forest modeling, feature engineering, and documentation.

Built by **Parul Chaudhary** · [LinkedIn](https://www.linkedin.com/in/parul-chaudhary-39269b213) · [Email](mailto:parul.jaswant@gmail.com)
