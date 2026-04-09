# 🏦 Loan Approval Prediction System
### 🎓 Machine Learning Final Exam Project

This repository contains a complete end-to-end Machine Learning pipeline designed to predict loan eligibility based on customer details. The project covers the full lifecycle from data cleaning to a live web deployment.

---

## 🚀 Project Overview
The goal of this project is to automate the loan eligibility process. It uses a **Random Forest Classifier** to determine if a loan should be "Approved" or "Rejected" based on features like credit history, income, and education.

- **Task Type:** Binary Classification
- **Dataset:** [Loan Prediction Dataset](https://www.kaggle.com/datasets/ninzaami/loan-predication)
- **Deployment Platform:** Hugging Face Spaces (Gradio)

---

## 🛠️ Machine Learning Pipeline
As per the exam requirements, the following steps were implemented:

1. **Data Loading:** Dataset loaded and verified with `.head()` and `.shape`.
2. **Data Preprocessing:** 
    *   Handled missing values using **Median/Mode Imputation**.
    *   Feature Engineering: Created `Total_Income` (Applicant + Co-applicant).
    *   **Scaling:** Applied `StandardScaler` to numerical features.
    *   **Encoding:** Applied `OneHotEncoder` to categorical features.
3. **Pipeline Construction:** Integrated preprocessing and model into a single Scikit-learn `Pipeline`.
4. **Model Selection:** Selected **Random Forest** for its robustness against outliers and ability to handle tabular data.
5. **Validation:** Performed **5-Fold Cross-Validation** (Avg Accuracy: ~80%).
6. **Hyperparameter Tuning:** Optimized using `GridSearchCV` to find the best `max_depth` and `n_estimators`.
7. **Evaluation:** Detailed analysis using **Confusion Matrix** and **Classification Report**.

---

## 📁 Repository Structure
| File | Description |
| :--- | :--- |
| `rf-train.py` | The main Python script for training and saving the model. |
| `app.py` | The Gradio script used for the web interface. |
| `loan_model.pkl` | The serialized best-performing model (Joblib format). |
| `requirements.txt` | List of required Python libraries for deployment. |
| `loan_predication.csv` | The dataset used for the project. |

---

## 💻 How to Run Locally

## How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Train the model:
python rf-train.py

3. Start Gradio App:
python app.py

## Links
- **Live Link**: https://huggingface.co/spaces/sohun-31/Loan-Approval-Prediction
