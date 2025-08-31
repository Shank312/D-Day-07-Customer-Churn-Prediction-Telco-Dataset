
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)

    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)

    print("Model Evaluation")
    print("Precision: ", round(precision, 3))
    print("Recall: ", round(recall, 3))
    print("F1 Score: ", round(f1, 3))

    return{"precision ": precision, "recall ": recall, "f1 ": f1 }