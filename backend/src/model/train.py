from xgboost import XGBClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
import mlflow

def train_model(X_train, y_train):
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("learning_rate", 0.1)

    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    print("\n=== Rapport de classification ===")
    print(report)
    print(f"\nBalanced accuracy score: {balanced_acc:.4f}")

    mlflow.log_metric("balanced_accuracy", balanced_acc)

    return report
