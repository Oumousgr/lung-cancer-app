import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template
from src.data.preprocess import load_and_prepare_data
from src.model.train import train_model, evaluate_model
from src.model.save_load import save_model, load_model
from src.model.predict import predict_from_json
import mlflow
import joblib

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "OK"})

@app.route("/train", methods=["GET"])
def train():
    try:
        with mlflow.start_run():
            X_train, X_test, y_train, y_test, scaler, columns = load_and_prepare_data()
            model = train_model(X_train, y_train)
            save_model(model, scaler, columns)
            report = evaluate_model(model, X_test, y_test)
            return jsonify({
                "message": "✅ Modèle entraîné avec succès.",
                "report": report
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        model, scaler, columns = load_model()
        input_data = request.get_json()

        print("📥 Données reçues dans /predict :", input_data)
        prediction = predict_from_json(input_data, model, scaler, columns)

        print("✅ Prédiction réalisée :", prediction.tolist())
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        print("❌ Erreur dans /predict :", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # ✅ Koyeb utilisera la variable d'environnement PORT
    app.run(host="0.0.0.0", port=port, debug=True)
