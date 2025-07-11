import sys
import os
import json

# Ajoute la racine du projet au path pour que app soit importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app import app

def test_predict():
    client = app.test_client()

    sample_input = [
        {
            "gender": "Male",
            "age": 60,
            "smoking_status": "Yes",
            "yellow_fingers": "Yes",
            "anxiety": "No",
            "peer_pressure": "Yes",
            "chronic_disease": "No",
            "fatigue": "Yes",
            "allergy": "No",
            "wheezing": "Yes",
            "alcohol_consuming": "Yes",
            "coughing": "Yes",
            "shortness_of_breath": "Yes",
            "swallowing_difficulty": "No",
            "chest_pain": "Yes",

            "family_history": "No",
            "hypertension": "No",
            "asthma": "No",
            "cirrhosis": "No",
            "other_cancer": "No",
            "treatment_type": "Chemotherapy",  # ou ce que tu as utilisé dans les données
            "country": "France",
            "cancer_stage": "Stage II"
        }
    ]

    response = client.post(
        "/predict",
        data=json.dumps(sample_input),
        content_type='application/json'
    )

    print("Response status:", response.status_code)
    print("Response data:", response.data.decode())

    assert response.status_code == 200
    assert "prediction" in response.json
    assert isinstance(response.json["prediction"], list)
