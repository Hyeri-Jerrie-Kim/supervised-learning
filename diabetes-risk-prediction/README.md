# 🩺 Diabetes Risk Prediction

This project demonstrates a complete machine learning pipeline to identify individuals at risk of diabetes using the **Pima Indians Diabetes dataset**. It was designed with real-world screening applications in mind—prioritizing **high recall**, **clinical interpretability**, and **model transparency** to support early diagnosis.

---

## 🎯 Objective

This project aims to build a **recall-optimized and interpretable model** that can flag potentially diabetic patients early, ensuring they are directed toward further medical testing. Rather than optimizing for accuracy, this model is tailored for **screening sensitivity**—minimizing false negatives in a healthcare context.

---

## 📊 Dataset Overview

- **Source**: [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Size**: 768 rows × 8 clinical features
- **Target**: `Outcome` (1 = diabetic, 0 = non-diabetic)
- **Note**: Some features (e.g., `Glucose`, `Insulin`) contain biologically implausible zeros, handled as missing values.

### About the Dataset

The dataset includes diagnostic data from 768 adult female Pima Indian patients.  
It is commonly used for educational and benchmarking purposes but not representative of a general population. This model is therefore a **proof of concept** demonstrating how recall-focused and explainable modeling can work in healthcare.

---

## 🧪 Approach Summary

### 🧹 Data Preprocessing

- Replaced implausible zeros using **median imputation**
- Applied **two-sided and one-sided outlier clipping** for stable training

### ⚙️ Model Development

- Built a **soft voting ensemble** combining:
  - Logistic Regression (tuned)
  - Decision Tree (tuned)
  - Random Forest
  - Extra Trees
  - SVM
- Used **SMOTE** to balance class distributions
- Optimized classification **threshold to 0.35** for higher recall

### 🔍 Model Interpretation

- Applied **SHAP** for:
  - Global feature importance (`summary_plot`)
  - Local patient-specific explanations (`force_plot`)
- Compared local values (e.g., `Glucose = 130`) against global feature distributions

---

## 📈 Results (Threshold = 0.35)

| Metric        | Value |
|---------------|--------|
| Recall (Class 1)   | 0.85   |
| Precision (Class 1) | 0.60   |
| F1 Score       | 0.70   |
| ROC AUC        | 0.84   |


> The selected threshold balances high recall and clinical trust, minimizing missed cases of diabetes.

---

## 🧠 Key Insights

- **Glucose**, **BMI**, and **Age** are the most influential predictors, aligning with clinical intuition.
- Model performance is **threshold-tuned**, **balanced**, and **interpretable**, supporting use in pre-screening scenarios.
- The final Jupyter notebook explains every step from data cleaning to prediction interpretation.

---

## 💡 Folder Structure
```
diabetes-risk-prediction/
├── notebooks/ 
│ └── diabetes_risk_prediction.ipynb
├── data/ 
│ └── processed/
│ └── raw/
│     └── pima-indians-diabetes.csv
├── images/ 
├── requirements.txt 
└── README.md 
```
---

## 🚀 Future Improvements

- Retrain the model on real-world, diverse population datasets
- Integrate time-series features for longitudinal patient tracking
- Evaluate fairness and subgroup performance (e.g., by age group)

---

## 📬 Contact Me

Hyeri Kim | 📧 [Email](mailto:hyeri5524@gmail.com) | 🌐 [LinkedIn](https://linkedin.com/in/hyerikim-ds)   
