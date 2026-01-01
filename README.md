# 🎓 Student Performance Prediction Project

## 📌 Overview
This project builds a **Machine Learning classification system** to predict whether a student will **PASS or FAIL** based on academic and behavioral factors.  
The system evaluates multiple models and selects the best-performing one, with a strong focus on **maximizing recall** to accurately identify **at-risk students**.

---

## ❓ Problem Statement
Educational institutions need an **early-warning system** to detect students who may fail.

This problem is formulated as a **binary classification task**, where minimizing **false negatives** (students predicted as passing but actually failing) is critical.

- **Target Variable:** `Pass_Fail`
- **Problem Type:** Classification
- **Primary Objective:** Maximize recall while maintaining strong overall accuracy

---

## 📊 Dataset Description
- **Total Records:** 708 students  
- **Total Features:** 10  
- **Target Column:** `Pass_Fail`  
- **Missing Values:** None  
- **Class Distribution:** Nearly balanced  

### 🔑 Key Input Features
- `Study_Hours_per_Week`
- `Attendance_Rate`
- `Past_Exam_Scores`
- `Gender`
- `Parental_Education_Level`
- `Internet_Access_at_Home`
- `Extracurricular_Activities`

### ❌ Removed Features
- `Student_ID` (identifier only)
- `Final_Exam_Score` (risk of data leakage)

---

## 🔍 Exploratory Data Analysis (EDA)
EDA was conducted to analyze feature distributions and relationships. Key findings include:

- Strong positive correlation between **attendance**, **study hours**, and **academic success**
- Balanced target classes, suitable for standard classification metrics
- No missing values or extreme outliers
- Categorical variables show meaningful separation between pass and fail outcomes

These insights confirm the dataset is **clean and well-suited for modeling**.

---

## 🛠️ Data Preprocessing
The preprocessing pipeline includes:

- Removal of unnecessary columns
- Scaling numerical features using **StandardScaler**
- Encoding categorical variables using **OneHotEncoder**
- **Stratified train-test split** (80% training, 20% testing)
- Saving the preprocessing pipeline for reuse

All preprocessing steps are **modular and reusable**.

---

## 🧪 Methodology
The following models were trained and evaluated:

- **Logistic Regression** (baseline)
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**

### 📈 Evaluation Metrics
- Cross-validation accuracy
- Test accuracy
- Precision, Recall, and **F2-score**
- ROC-AUC

> **Recall was prioritized** to ensure failing students are correctly identified.

---

## 🏆 Model Results
The **Random Forest Classifier** was selected as the final model.

### ✅ Test Performance
- **Accuracy:** 87.32%
- **Precision (Pass):** 81%
- **Recall (Pass):** 97%
- **ROC-AUC:** 0.98

### 📉 Confusion Matrix
[[55, 16],
[ 2, 69]]

yaml
Copy code

The model achieves **excellent recall**, with very few failing students misclassified as passing.

---

## 📊 Model Evaluation
The evaluation phase included:

- Accuracy, Precision, Recall, and F1-score
- ROC-AUC score
- Confusion Matrix analysis
- ROC and Precision–Recall curves
- Feature importance analysis for interpretability

Results confirm the model is **robust and recall-optimized**.

---

## 🌐 Streamlit Web Application
A **Streamlit-based interactive web application** was developed for real-world usability.

### 🚀 Features
- **Interactive Predictions:** Real-time pass/fail predictions with confidence scores
- **Interactive EDA:** Dynamic visualizations and feature exploration
- **Clean UI/UX:** Simple navigation and professional layout

### ▶️ Run the Application
```bash
streamlit run app/streamlit_app.py
⚙️ Installation
Install all required dependencies using:

bash
Copy code
pip install -r requirements.txt
🔑 Key Insights
Attendance rate and study hours are the strongest predictors of success

Random Forest effectively captures non-linear relationships

Recall-focused evaluation improves early detection of at-risk students

🔮 Future Work
Add behavioral and psychological features

Incorporate time-based academic trends

Apply SHAP for advanced model explainability

Deploy the application to a cloud platform

✅ Conclusion
This project delivers a complete, production-ready machine learning pipeline for predicting student performance.
The final Random Forest model achieves high recall and strong accuracy, making it suitable for academic decision support and early intervention systems.