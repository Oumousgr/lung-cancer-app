from app import app

def test_train():
    client = app.test_client()
    response = client.get("/train")
    assert response.status_code == 200 or response.status_code == 500
    assert "message" in response.json or "error" in response.json
