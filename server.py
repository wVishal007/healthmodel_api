from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from fastapi.middleware.cors import CORSMiddleware

# ------------------------
# 1. Initialize FastAPI
# ------------------------
app = FastAPI()

# Allow frontend access
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://disease-predict-two.vercel.app"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# 2. Load trained model + encoders
# ------------------------
model_DISEASE = joblib.load("DiseasePredictModel.joblib")
model_OE = joblib.load("encoder.joblib")
model_LE = joblib.load("label_encoder.joblib")

# Define columns
categorical_cols = ["Fever", "Cough", "Fatigue", "Difficulty Breathing",
                    "Gender", "Blood Pressure", "Cholesterol Level"]
numeric_cols = ["Age"]

# ------------------------
# 3. Response model
# ------------------------
class PredictionResponse(BaseModel):
    prediction: str

@app.get("/")
def home():
    return {"message": "Disease Prediction API is running 🚀"}

# ------------------------
# 4. Prediction Endpoint
# ------------------------
# ------------------------
# 3. Response model (updated)
# ------------------------
class PredictionResponse(BaseModel):
    top_diseases: list


@app.post("/predict", response_model=PredictionResponse)
def predict(symptoms: dict):
    """
    Example input:
    {
        "Fever": "Yes",
        "Cough": "No",
        "Fatigue": "Yes",
        "Difficulty Breathing": "Yes",
        "Age": 25,
        "Gender": "Male",
        "Blood Pressure": "Normal",
        "Cholesterol Level": "Normal"
    }
    """
    input_df = pd.DataFrame([symptoms])

    # Encode categorical + keep numeric
    encoded_cats = model_OE.transform(input_df[categorical_cols]).toarray()
    numerics = input_df[numeric_cols].values

    # Merge back
    input_encoded = np.concatenate([encoded_cats, numerics], axis=1)

    # Predict probabilities for all classes
    probs = model_DISEASE.predict_proba(input_encoded)[0]

    # Get top 3 classes
    top_indices = np.argsort(probs)[-3:][::-1]  # highest 3
    top_labels = model_LE.inverse_transform(top_indices)
    top_probs = probs[top_indices]

    # Combine into a nice list
    top_diseases = [
        {"disease": label, "probability": float(f"{prob*100:.2f}")}
        for label, prob in zip(top_labels, top_probs)
    ]

    return {"top_diseases": top_diseases}
