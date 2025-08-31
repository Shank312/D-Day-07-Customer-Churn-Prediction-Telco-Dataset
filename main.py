
import pandas as pd
from data_loader import load_data
from preprocess import preprocess_data
from data_splitter import split_data
from logistic_trainer import train_logistic
from random_forest_trainer import train_random_forest
from xgboost_trainer import train_xgboost
from evaluator import evaluate_model
from model_exporter import save_model

if __name__ == "__main__":
    # Load + preprocess
    df = load_data("Telco_customer_churn.csv")
    df = preprocess_data(df)

    # Train-test split
    X_train, X_test, Y_train, Y_test = split_data(df)

    # Train models
    log_model = train_logistic(X_train, Y_train)
    rf_model = train_random_forest(X_train, Y_train)
    xgb_model = train_xgboost(X_train, Y_train)

    # Evaluate 
    print("\n Logistic Regression: ")
    evaluate_model(log_model, X_test, Y_test)

    print("\n Random Forest: ")
    evaluate_model(rf_model, X_test, Y_test)

    print("\n XGBoost: ")
    evaluate_model(xgb_model, X_test, Y_test)

    # Save best model (example: XGBoost)
    save_model(xgb_model, "churn_model.pkl")