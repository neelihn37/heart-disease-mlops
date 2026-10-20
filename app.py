from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

#app = FastAPI(title="Heart Disease Prediction API", version="1.0")

from fastapi import FastAPI

app = FastAPI()  # <-- Must be named 'app' to match 'app:app'

@app.get("/")
def read_root():
    return {"message": "Heart Disease MLOps API is running"}


# Load your trained model (make sure your model file is copied/available)
# model = joblib.load("models/best_model.pkl")

# Define input data structure based on UCI Heart Disease features
class HeartDiseaseFeatures(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/")
def read_root():
    return {"message": "Welcome to the Heart Disease Prediction API!"}

@app.post("/predict")
def predict_heart_disease(features: HeartDiseaseFeatures):
    try:
        # Convert input to DataFrame for the model
        input_data = pd.DataFrame([features.dict()])
        
        # Dummy prediction response for testing if model isn't wired yet, 
        # or replace with: prediction = model.predict(input_data)
        prediction = 1  # Example output
        probability = 0.85
        
        return {
            "prediction": int(prediction),
            "heart_disease_probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))