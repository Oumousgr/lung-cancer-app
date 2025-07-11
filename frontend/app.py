import streamlit as st
import requests
from datetime import date

st.title("🫁 Prédiction de Survie au Cancer du Poumon")

API_URL = "http://127.0.0.1:5001/predict"  # Backend Flask URL

# --- Saisie des données utilisateur ---
age = st.number_input("Âge", min_value=0.0, max_value=120.0, value=60.0)
gender = st.selectbox("Genre", ["Male", "Female"])
country = st.text_input("Pays", "France")
diagnosis_date = st.date_input("Date de diagnostic", value=date(2023, 1, 1))
cancer_stage = st.selectbox("Stade du cancer", ["Stage I", "Stage II", "Stage III", "Stage IV"])
family_history = st.selectbox("Antécédents familiaux", ["Yes", "No"])
smoking_status = st.selectbox("Statut tabagique", ["Never Smoked", "Former Smoker", "Passive Smoker", "Active Smoker"])
bmi = st.number_input("IMC", min_value=10.0, max_value=60.0, value=25.0)
cholesterol_level = st.number_input("Niveau de cholestérol", min_value=100, max_value=500, value=200)
hypertension = st.selectbox("Hypertension ?", ["Yes", "No"])
asthma = st.selectbox("Asthme ?", ["Yes", "No"])
cirrhosis = st.selectbox("Cirrhose ?", ["Yes", "No"])
other_cancer = st.selectbox("Autre cancer ?", ["Yes", "No"])
treatment_type = st.selectbox("Type de traitement", ["Chemotherapy", "Surgery", "Radiation", "Combined"])
end_treatment_date = st.date_input("Date fin de traitement", value=date(2024, 1, 1))

# --- Encodage simple ---
def yes_no(val): return 1 if val == "Yes" else 0

if st.button("Prédire la survie"):
    data = {
        "age": age,
        "gender": gender,
        "country": country,
        "diagnosis_date": diagnosis_date.isoformat(),
        "cancer_stage": cancer_stage,
        "family_history": yes_no(family_history),
        "smoking_status": smoking_status,
        "bmi": bmi,
        "cholesterol_level": cholesterol_level,
        "hypertension": yes_no(hypertension),
        "asthma": yes_no(asthma),
        "cirrhosis": yes_no(cirrhosis),
        "other_cancer": yes_no(other_cancer),
        "treatment_type": treatment_type,
        "end_treatment_date": end_treatment_date.isoformat()
    }

    try:
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            result = response.json()
            if result["prediction"][0] == 1:
                st.success("✅ Le patient a survécu.")
            else:
                st.error("⚠️ Le patient n'a pas survécu.")
        else:
            st.error(f"Erreur serveur (code {response.status_code})")
    except Exception as e:
        st.error(f"Erreur de connexion au backend : {e}")
