
import joblib

def save_model(model, filename="churn_model.pkl"):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")