from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import joblib

app = FastAPI()

# Load the sepsis prediction model
model = joblib.load('XGB.joblib')

@app.get("/")
async def read_root():
    return {"message": "Sepsis Prediction API using FastAPI"}

def classify(prediction):
    if prediction == 0:
        return "Patient does not have sepsis"
    else:
        return "Patient has sepsis"

@app.get("/predict/")
async def predict_sepsis(
    prg: float = Query(..., description="Plasma glucose"),
    pl: float = Query(..., description="Blood Work Result-1 (mu U/ml)"),
    pr: float = Query(..., description="Blood Pressure (mm Hg)"),
    sk: float = Query(..., description="Blood Work Result-2 (mm)"),
    ts: float = Query(..., description="Blood Work Result-3 (mu U/ml)"),
    m11: float = Query(..., description="Body mass index (weight in kg/(height in m)^2"),
    bd2: float = Query(..., description="Blood Work Result-4 (mu U/ml)"),
    age: int = Query(..., description="Patient's age (years)")
):
    input_data = [prg, pl, pr, sk, ts, m11, bd2, age]

    input_df = pd.DataFrame([input_data], columns=[
        "Plasma glucose", "Blood Work Result-1", "Blood Pressure",
        "Blood Work Result-2", "Blood Work Result-3",
        "Body mass index", "Blood Work Result-4", "Age"
    ])

    pred = model.predict(input_df)
    output = classify(pred[0])

    response = {
        "prediction": output
    }

    return response

# Run the app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860)
