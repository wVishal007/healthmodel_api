from fastapi import FastAPI
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# Init FastAPI
app = FastAPI()


origins = [
    "https://disease-predict-two.vercel.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # list of allowed origins, e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],    # GET, POST, PUT, DELETE, OPTIONS
    allow_headers=["*"],    # allow all headers
)

# Load encoders and model
model_OE = joblib.load('encoder.joblib')
model_LE = joblib.load('label_encoder.joblib')
model_DISEASE = joblib.load('DiseasePredictModel.joblib')

# Get class names from label encoder
class_names = model_LE.classes_



@app.get("/")
def home():
    return {"message": "Disease Prediction API is running 🚀"}

@app.post("/predict")
def predict(symptoms: dict):
    """
    Example input (JSON):
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

    # Transform input
    input_encoded = model_OE.transform(input_df).toarray()

    # Predict
    prediction = model_DISEASE.predict(input_encoded)[0]

    # Decode label
    predicted_label = model_LE.inverse_transform([prediction])[0]

    return {"prediction": predicted_label}
