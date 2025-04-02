# 🩺 Diabetes Risk Prediction

📌 **Author:** Hyeri Kim  
📅 **Last Updated:** April 2025  
📂 **Category:** Data Science, Machine Learning, Healthcare Analytics  
🗂 **Dataset:** [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  

---

## 📖 Project Overview

This project leverages ensemble learning techniques to predict the likelihood of diabetes using patient data. By combining multiple machine learning models, the solution serves as a robust screening tool, enabling early detection and flagging high-risk individuals for further clinical evaluation. The codebase has been modularized into dedicated Python files for data preprocessing, model training, evaluation, hyperparameter tuning, and inference, ensuring a clean, maintainable, and scalable workflow.

---

## 📌 Key Features

- 🔍 **Data Preprocessing:**  
  Loaded and cleaned the raw dataset by handling implausible values and imputing missing data using median values. Split the data into training and testing sets with stratification.

- 🧠 **Ensemble Modeling:**  
  Implemented multiple base classifiers (Logistic Regression, Random Forest, SVM, Decision Tree, Extra Trees) and combined them into a soft voting ensemble to enhance predictive performance on imbalanced data.

- 📊 **Comprehensive Evaluation:**  
  Generated detailed evaluation metrics including classification reports, confusion matrices, ROC curves, and Precision-Recall curves. Visualized feature importance to interpret key predictors.

- 🎯 **Hyperparameter Tuning:**  
  Used GridSearchCV to optimize the regularization parameter for Logistic Regression, ensuring an optimal balance between model complexity and performance.

- 🛠 **Modularized Code Structure:**  
  Divided the workflow into clear, reusable modules to improve maintainability and facilitate collaboration.

---

## 📊 Dataset

- **📍 Source:** [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **📁 File:** `pima-indians-diabetes.csv`
- **🔢 Total Rows:** 768  
- **🔑 Key Columns:**
  - `Pregnancies` - Number of pregnancies  
  - `Glucose` - Plasma glucose concentration  
  - `BloodPressure` - Diastolic blood pressure (mm Hg)  
  - `SkinThickness` - Triceps skin fold thickness (mm)  
  - `Insulin` - 2-Hour serum insulin (mu U/ml)  
  - `BMI` - Body mass index (weight in kg/(height in m)^2)  
  - `DiabetesPedigreeFunction` - Diabetes pedigree function  
  - `Age` - Age in years  
  - `Outcome` - Class variable (0 = non-diabetic, 1 = diabetic)

### About the Dataset

The Pima Indians Diabetes dataset contains diagnostic measurements for 768 female patients of Pima Indian heritage. It is widely used for benchmarking classification algorithms in healthcare analytics.

---

## 🚀 Technologies Used

- **Python** 🐍 (Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Joblib)
- **Jupyter Notebook** 📒
- **Git** for version control

---

## 🔧 Setup & Installation

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/Hyeri-Jerrie-Kim/diabetes-risk-prediction.git
   cd diabetes-risk-prediction
   ```
2. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Launch the Jupyter Notebook:**
   ```sh
   jupyter notebook
   ```
4. Run the Notebook: Open the `diabetes-risk-prediction.ipynb` file located in the notebooks/ folder to explore the analysis.

---

## 📊 Key Insights & Results

### 1. Model Performance
- **Cross-Validation (Training Data):**  
  Base models yielded F1-scores (mean ± std) ranging from 0.56 to 0.67, with Logistic Regression, SVM, and Extra Trees performing best.
  
- **Ensemble Performance on Test Data:**  
  - **Precision (Diabetic):** ~0.70  
  - **Recall (Diabetic):** ~0.62  
  - **F1-Score (Diabetic):** ~0.66  
  - **Accuracy:** ~0.77  
  - **AUC (ROC):** ~0.83

### 2. Error Analysis
- **Confusion Matrix Highlights:**  
  - True Negatives: 129  
  - False Positives: 21  
  - False Negatives: 31  
  - True Positives: 50  
  *Note:* The presence of 31 false negatives underscores the need for further improvement in recall for this critical healthcare application.

### 3. Feature Importance
- **Key Predictors:**  
  The ensemble model identifies **Glucose** as the most influential feature, followed by **BMI** and **Age**—consistent with established clinical risk factors for diabetes.

### 4. Hyperparameter Tuning
- **Optimized Logistic Regression:**  
  GridSearchCV determined that `C=1` is optimal, yielding a cross-validated F1-score of approximately 0.6775. This result confirms that moderate regularization effectively balances model complexity and performance.

---

## 📜 Future Improvements

- **Enhance Recall:**  
  Adjust decision thresholds or implement resampling techniques (e.g., SMOTE) to further reduce false negatives.
  
- **Increase Model Interpretability:**  
  Integrate advanced interpretability tools such as SHAP or LIME to provide detailed insights into individual predictions.
  
- **Continuous Monitoring:**  
  Develop a pipeline for ongoing data ingestion and model monitoring to adapt to evolving patient demographics and risk factors.

---

## 📬 Contact Me

📧 [Hyeri Kim](mailto:hyeri5524@gmail.com) | 🌐 [LinkedIn](https://linkedin.com/in/hyerikim-ds)
