
from sklearn.linear_model import LogisticRegression

def train_logistic(X_train, Y_train):
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, Y_train)
    print("Logistic Regression Trained")
    return log_model