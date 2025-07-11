import pandas as pd
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

def load_and_prepare_data(csv_path="data/raw/LungCancer.csv", target_column="survived"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Le fichier '{csv_path}' n'a pas été trouvé.")
    
    data = pd.read_csv(csv_path)

    categorical_columns = ['gender', 'smoking_status', 'family_history', 'hypertension',
                           'asthma', 'cirrhosis', 'other_cancer', 'treatment_type',
                           'country', 'cancer_stage']
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

    return X_train_scaled, X_test_scaled, y_train_res, y_test, scaler, X.columns.tolist()

print("Fichier preprocess.py bien exécuté ✅")

print("Fonctions dans ce module :", dir())
