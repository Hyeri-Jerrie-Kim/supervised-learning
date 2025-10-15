# 🩺 Diabetes Risk Prediction

## 🏷 Project Overview
This project predicts diabetes risk using clinical data to support early screening decisions. It reflects my growth from data analyst to data scientist — integrating model development, tuning, and explainability to extract meaningful healthcare insights.

**Dataset**: [pima indians diabetes database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

## 🎯 Objectives
Diabetes is often underdiagnosed until symptoms progress. The goal of this project is to identify high-risk individuals earlier through data-driven pre-screening.  
Because missing potential cases can delay treatment, **recall** was prioritized to minimize false negatives while maintaining acceptable precision.

## 🧩 Workflow Summary
- The modeling process began with a baseline **Logistic Regression** and an **ensemble model** without oversampling. Since recall showed little improvement (0.53 → 0.54), **SMOTE** was applied to address class imbalance.  
- After retraining with SMOTE, recall increased to 0.73 while AUC remained stable (~0.83).  
- Further **threshold tuning (0.5 → 0.4 → 0.35 → 0.30)** identified **0.30** as the optimal decision point — maximizing recall (0.88) and F1-score (0.71) with minimal precision loss.  
- This final threshold balances medical sensitivity and model reliability for pre-screening use.

## 💡 Key Results
- The final model achieved **Recall = 0.88** and **ROC-AUC = 0.83**, showing strong sensitivity and stable discriminative power.  
- While precision decreased slightly (0.59), this trade-off reduced false negatives to only 12% — a crucial improvement in a clinical screening context.  
- These results demonstrate how model calibration can align machine learning with healthcare priorities.

## 🩺 Explainability (SHAP Insights)
- SHAP analysis revealed **Glucose, BMI, Age, and Insulin** as the most influential features.  
High glucose and BMI values consistently increased diabetes probability, aligning with medical evidence.  
- This interpretability not only enhances transparency but also validates that the model’s reasoning matches clinical logic.

## 🧠 Key Takeaways & Future Work
- I built an end-to-end classification pipeline that prioritizes recall and interpretability — key elements for healthcare AI.
- Through this project, I learned to evaluate models beyond accuracy, focusing on decision thresholds and explainability.
- Future improvements may include integrating more demographic data and deploying the model as a web-based screening tool.

## ⚙️ Tech Stack
Python | Pandas • NumPy • Scikit-learn | Matplotlib • Seaborn | SHAP (Explainable AI)  

## 🚀 How to Run
Clone the repository and install dependencies:
```bash
git clone https://github.com/Hyeri-Jerrie-Kim/supervised-learning.git
cd diabetes-risk-prediction

pip install -r requirements.txt
```
Run the notebook:
```bash
jupyter notebook notebooks/diabetes_risk_prediction.ipynb
```

## 📬 Contact Me

Hyeri Kim | 📧 [Email](mailto:hyeri5524@gmail.com) | 🌐 [LinkedIn](https://linkedin.com/in/hyerikim-ds)   
