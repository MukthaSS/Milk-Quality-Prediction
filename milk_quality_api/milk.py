from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# -----------------------------
# Load trained model
# -----------------------------
with open("milknew[1].sav", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Milk Quality Prediction API",
    description="Predicts Milk Quality: Low, Medium, High",
    version="1.0"
)

# -----------------------------
# Input schema (STRICT)
# -----------------------------
class MilkInput(BaseModel):
    pH: float
    temperature: float
    taste: int
    odor: int
    fat: int
    turbidity: int
    colour: int

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def health_check():
    return {"status": "Milk Quality API is running"}

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_milk_quality(data: MilkInput):

    # Convert input to numpy array (same order as training)
    features = np.array([
        data.pH,
        data.temperature,
        data.taste,
        data.odor,
        data.fat,
        data.turbidity,
        data.colour
    ]).reshape(1, -1)

    prediction = model.predict(features)[0]

    if prediction == 3:
        grade = "High"
    elif prediction == 2:
        grade = "Medium"
    else:
        grade = "Low"

    return {
        "prediction_class": int(prediction),
        "milk_quality": grade
    }
