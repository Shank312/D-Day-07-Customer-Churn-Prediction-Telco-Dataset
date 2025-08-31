
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, Y_train):
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_model.fit(X_train, Y_train)
    print("Random Forest Trained")
    return rf_model