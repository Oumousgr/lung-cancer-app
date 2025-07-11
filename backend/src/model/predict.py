import pandas as pd
from datetime import datetime

def predict_from_json(input_data, model, scaler, column_names):
    # Convertir les données reçues en DataFrame
    input_df = pd.DataFrame([input_data])  # <- Très important : mettre dans une liste pour créer une ligne

    # Supprimer l'ID si présent
    if 'id' in input_df.columns:
        input_df = input_df.drop('id', axis=1)

    # Colonnes catégorielles à encoder
    categorical_columns = [
        'gender', 'smoking_status', 'family_history', 'hypertension',
        'asthma', 'cirrhosis', 'other_cancer', 'treatment_type',
        'country', 'cancer_stage'
    ]
    input_df = pd.get_dummies(input_df, columns=categorical_columns)

    # Ajouter les colonnes manquantes avec valeur 0
    missing_cols = set(column_names) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0

    # Réordonner les colonnes
    input_df = input_df[column_names]

    # Traitement des dates
    reference_date = datetime(2000, 1, 1)

    input_df['diagnosis_date'] = pd.to_datetime(input_df['diagnosis_date'], errors='coerce')
    input_df['diagnosis_date'] = (input_df['diagnosis_date'] - reference_date).dt.days

    input_df['end_treatment_date'] = pd.to_datetime(input_df['end_treatment_date'], errors='coerce')
    input_df['end_treatment_date'] = (input_df['end_treatment_date'] - reference_date).dt.days

    # Appliquer le scaler
    input_scaled = scaler.transform(input_df)

    # Faire la prédiction
    prediction = model.predict(input_scaled)

    return prediction
