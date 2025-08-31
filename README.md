📊 Customer Churn Prediction (Telco Dataset)

This project predicts customer churn for a telecom company using machine learning models.
The dataset contains customer demographics, services, contract details, and billing information.
The goal is to predict whether a customer will churn (leave) or stay.


🚀 Project Workflow

Load Dataset

CSV: Telco_customer_churn.csv

Handled using pandas.

Preprocessing

Dropped non-informative columns (CustomerID, redundant churn columns).

Converted Total Charges to numeric.

Encoded target column Churn → (Yes=1, No=0).

One-hot encoded categorical columns.

Final dataset: 7043 customers × 2816 features.

Train-Test Split

80% training, 20% testing.

Stratified split to preserve churn ratio.

Models Trained

Logistic Regression (baseline linear classifier).

Random Forest (ensemble of decision trees).

XGBoost (boosting algorithm, high performance).

Evaluation Metrics

Precision → How many predicted churners are real churners.

Recall → How many real churners are caught.

F1-Score → Balance between precision & recall.

| Model               | Precision | Recall | F1 Score |
| ------------------- | --------- | ------ | -------- |
| Logistic Regression | 0.836     | 0.845  | 0.840    |
| Random Forest       | 0.896     | 0.805  | 0.848    |
| XGBoost             | 0.851     | 0.872  | 0.861    |


✅ XGBoost performed best.

Save Model

Best model saved as: churn_model.pkl (using joblib).


📂 Project Structure
03_customer_churn/
│── data_loader.py          # Load dataset
│── preprocess.py           # Preprocess categorical & numerical features
│── data_splitter.py        # Train-test split
│── logistic_trainer.py     # Train Logistic Regression
│── random_forest_trainer.py# Train Random Forest
│── xgboost_trainer.py      # Train XGBoost
│── evaluator.py            # Evaluate models (Precision, Recall, F1)
│── model_exporter.py       # Save trained model
│── main.py                 # Full pipeline runner
│── churn_model.pkl         # Saved best model (XGBoost)
│── Telco_customer_churn.csv# Dataset
│── README.md               # Project documentation



⚡ How to Run

Clone repository or open project folder.

Install dependencies:
pip install pandas numpy scikit-learn xgboost joblib

Run full pipeline:
python main.py


🎯 Key Learnings

Churn prediction is critical for customer retention in telecom.

XGBoost outperforms traditional models for churn.

Precision/Recall trade-off matters more than accuracy for imbalanced data.


🔮 Next Steps

Feature importance analysis (which features drive churn).

Hyperparameter tuning for XGBoost.

Handle class imbalance (SMOTE, class weights).

Deploy model as an API (Flask/FastAPI).


👨‍💻 Author: Shankar Kumar
📌 AI/ML Mastery Path — Day 07 Project
