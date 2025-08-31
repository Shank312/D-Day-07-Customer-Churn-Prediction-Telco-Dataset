
from xgboost import XGBClassifier

def train_xgboost(X_train, Y_train):
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    xgb_model.fit(X_train, Y_train)
    print("XGBoost Trained")
    return xgb_model