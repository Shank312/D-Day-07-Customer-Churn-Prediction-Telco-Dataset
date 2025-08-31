
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    # Drop CustomerID (not useful)
    df = df.drop("CustomerID", axis=1)

    # Convert TotalCharges to numeric
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors = "coerce")

    df["Total Charges"] = df["Total Charges"].fillna(0)

   

    # Encode target column
    df["Churn"] = df["Churn Label"].map({"Yes": 1, "No": 0})

    # Drop redundant columns
    df = df.drop(["Churn Label", "Churn Reason", "Churn Value"], axis=1)

    # One - hot encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    print("Preprocessing Done. Shape: ", df.shape)
    print("Churn Distribution after preprocessing:\n", df["Churn"].value_counts())
    return df

if __name__ == "__main__":
    data = pd.read_csv("Telco_customer_churn.csv")
    processed = preprocess_data(data)
    print(processed.head())

