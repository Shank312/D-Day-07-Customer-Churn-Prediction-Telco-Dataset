
import pandas as pd

def load_data(path="Telco_customer_churn.csv"):
    df = pd.read_csv(path)
    print("Dataset Loaded. Shape: ", df.shape)
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
