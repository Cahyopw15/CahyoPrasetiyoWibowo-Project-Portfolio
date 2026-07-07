# Credit Risk Prediction Model

An end-to-end machine learning project aimed at predicting credit risk (Good vs. Bad Loans) using historical lending data. This project explores customer demographics, financial behaviors, loan characteristics, and evaluates multiple machine learning algorithms to identify the most robust model for deployment.

## 📌 Project Overview
Evaluating credit risk accurately is crucial for financial institutions to minimize non-performing loans (NPLs) while maximizing profitable lending. This repository covers:
* **Exploratory Data Analysis (EDA)** on loan features.
* **Data Preprocessing** (handling missing values, handling class imbalance, encoding, scaling).
* **Model Training & Hyperparameter Tuning** (Logistic Regression, Random Forest, XGBoost).

---

## 📊 Key Insights from EDA

### 1. Risk Drivers (Borrower Profile)
* **Interest Rate & DTI:** High interest rates (median ~16%) and elevated Debt-to-Income ratios (median ~18%) are the strongest indicators of loan default (*Bad Loans*).
* **Income:** Borrowers with a "Good Loan" status exhibit a wider, higher income range. "Bad Loan" status is highly concentrated among lower-income brackets.

### 2. Loan Distribution
* **Risk Grades:** Loans are heavily dominated by Grade B (29.39%) and Grade C (26.88%). Lower-risk grades (A and B) maintain very low default rates.
* **Loan Term:** Shorter-term loans (**36 months**) are highly preferred (72.46%) compared to 60-month terms. However, 60-month loans carry a disproportionately higher concentration of defaults.
* **Purpose:** **Debt Consolidation** (58.83%) and **Credit Cards** (22.36%) account for over 80% of all loan applications.

---

## 🤖 Model Evaluation & Hyperparameter Tuning

Initially, the models suffered from standard baseline issues:
* **Logistic Regression:** Failed on minority classes (0.00 Precision/Recall).
* **Random Forest:** Suffered from extreme overfitting (1.00 Train Accuracy vs. 0.81 Test Accuracy).

### Post-Hyperparameter Tuning Results
After tuning, all models effectively transitioned into a **Good Fit** status. 

| Model | Train Accuracy | Test Accuracy | Test ROC-AUC | Precision | Recall | F1 Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.8111 | 0.8114 | 0.6210 | 0.4853 | 0.0038 | 0.0076 | Good Fit |
| Random Forest | 0.8136 | 0.8114 | 0.7028 | 0.4957 | 0.0066 | 0.0131 | Good Fit |
| **XGBoost (Best Model)** | **0.8204** | **0.8117** | **0.7047** | **0.5060** | **0.0691** | **0.1216** | **Good Fit** |

### 🏆 Why XGBoost was chosen as the Best Model:
1. **Stable Generalization:** Maintains highly consistent metrics with a minimal gap between train and test accuracy (~82% vs ~81.17%).
2. **Superior Discrimination Power:** Achieved the highest **Test ROC-AUC (0.7047)**, making it the best model at distinguishing between defaulting and non-defaulting borrowers.
3. **Better Risk Capture:** Outperformed other models by capturing the highest **Recall (0.0691)** and **F1-Score (0.1216)** on the minority class.

---

## Recommendation
1. Operational Solutions (Risk Management)
DTI Cap & Term Filters: Limit the maximum Debt-to-Income (DTI) ratio to 15% for low-income borrowers and tighten credit scoring thresholds specifically for high-risk 60-month terms.

Direct Pay Feature: For debt consolidation loans (>58% of cases), disburse funds directly to the destination financial institutions instead of giving cash to the borrower to prevent misuse.

2. Technological Innovations (Data & Modeling)
Alternative Data Integration: Feed alternative data (e-commerce behavior, utility bills, digital wallets) into the XGBoost model to significantly improve its default-detection sensitivity (Recall).

Automated Decision Engine: Deploy the trained XGBoost model as a real-time, automated approval system to speed up preliminary risk screening.

3. Strategic Business Recommendations (Stakeholders)
Target Grade A/B Borrowers: Launch premium, low-interest products tailored to Grade A and B segments, as they have proven to have the lowest default rates.

Risk-Based Provisioning (IFRS 9): Utilize the model's default probability outputs to calculate the company's Expected Credit Loss (ECL) and optimize financial reserve allocations.

---


## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, XGBoost, Jupyter Notebook
