import sys
import os

# Ajoute la racine du projet au path pour que app soit importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app import app

def test_health():
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json == {"status": "OK"}
