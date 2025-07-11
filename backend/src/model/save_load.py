import joblib
from src.config import MODEL_PATH, SCALER_PATH, COLUMNS_PATH

def save_model(model, scaler, column_names):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(column_names, COLUMNS_PATH)

def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, scaler, columns
