from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
import kagglehub
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, balanced_accuracy_score
import joblib
from datetime import datetime
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, precision_score

# MLflow
import mlflow
from dotenv import load_dotenv

# Init Flask
app = Flask(__name__)

# Charger les variables d'environnement
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))

def download_data():
    path = kagglehub.dataset_download("khwaishsaxena/lung-cancer-dataset")
    return path

def load_and_prepare_data():
    dataset_path = os.path.join("data", "raw", "LungCancer.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Le fichier 'LungCancer.csv' n'a pas été trouvé à l'emplacement {dataset_path}")
    
    data = pd.read_csv(dataset_path)
    target_column = 'survived'

    categorical_columns = ['gender', 'smoking_status', 'family_history', 'hypertension', 'asthma', 
                           'cirrhosis', 'other_cancer', 'treatment_type', 'country', 'cancer_stage']
    data = pd.get_dummies(data, columns=categorical_columns)

    data['diagnosis_date'] = pd.to_datetime(data['diagnosis_date'], errors='coerce')
    data['end_treatment_date'] = pd.to_datetime(data['end_treatment_date'], errors='coerce')
    reference_date = datetime(2000, 1, 1)
    data['diagnosis_date'] = (data['diagnosis_date'] - reference_date).dt.days
    data['end_treatment_date'] = (data['end_treatment_date'] - reference_date).dt.days

    X = data.drop(target_column, axis=1)
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train_res, y_test, X_test_scaled, y_test, scaler, X.columns.tolist()

def train_model(X_train, y_train):
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=1
    )

    with mlflow.start_run():
        mlflow.log_params({
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "scale_pos_weight": 1,
            "model_type": "XGBoost"
        })
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "lung_cancer_model")

    return model

def save_model(model, scaler, column_names):
    joblib.dump(model, 'lung_cancer_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(column_names, 'columns.pkl')

def load_model():
    model = joblib.load('lung_cancer_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    # Log dans MLflow
    mlflow.log_metric("balanced_accuracy", balanced_acc)

    print("\n=== Rapport de classification ===")
    print(report)
    print(f"\nBalanced accuracy score: {balanced_acc:.4f}")
    return report

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['GET'])
def train():
    try:
        X_train, X_test, y_train, y_test, X_test_final, y_test_final, scaler, column_names = load_and_prepare_data()
        model = train_model(X_train, y_train)
        save_model(model, scaler, column_names)
        evaluation = evaluate_model(model, X_test_final, y_test_final)
        return jsonify({"message": "Modèle entraîné avec succès", "evaluation": evaluation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model, scaler = load_model()
        if os.path.exists('columns.pkl'):
            column_names = joblib.load('columns.pkl')
        else:
            raise FileNotFoundError("Le fichier 'columns.pkl' n'existe pas. Veuillez entraîner le modèle d'abord.")

        input_data = request.get_json()
        input_df = pd.DataFrame(input_data)

        if 'id' in input_df.columns:
            input_df = input_df.drop('id', axis=1)

        categorical_columns = ['gender', 'smoking_status', 'family_history', 'hypertension', 'asthma', 
                               'cirrhosis', 'other_cancer', 'treatment_type', 'country', 'cancer_stage']
        input_df = pd.get_dummies(input_df, columns=categorical_columns)

        missing_cols = set(column_names) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        input_df = input_df[column_names]

        input_df['diagnosis_date'] = pd.to_datetime(input_df['diagnosis_date'], errors='coerce')
        reference_date = datetime(2000, 1, 1)
        input_df['diagnosis_date'] = (input_df['diagnosis_date'] - reference_date).dt.days
        input_df['end_treatment_date'] = pd.to_datetime(input_df['end_treatment_date'], errors='coerce')
        input_df['end_treatment_date'] = (input_df['end_treatment_date'] - reference_date).dt.days

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)

        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
