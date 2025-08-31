
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df):
    X = df.drop("Churn", axis = 1)
    Y = df["Churn"]

    X_train, X_test, Y_train, Y_test = train_test_split (
        X, Y, test_size = 0.2, random_state=42, stratify = Y
    )

    print("Data Split Done")
    print("Training Set: ", X_train.shape, "Test Set: ", X_test.shape)
    return X_train, X_test, Y_train, Y_test

if __name__== "__main__":
    # load dataset
    df = pd.read_csv("Telco_customer_churn.csv")
    print("Columns in dataset: ", df.columns)

    # call function
    X_train, X_test, Y_train, Y_test = split_data(df)