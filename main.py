from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import json
import joblib

app = FastAPI()

# Load the XGBoost model
xgb_model = joblib.load("xgb_model.joblib")

@app.get("/")
async def read_root():
    return {"message": "Welcome to my Sepsis-Prediction-APP-using-FastAPI"}

@app.post("/predict/")
async def predict_sepsis(request: Request):
    try:
        # Get request JSON body
        request_data = await request.json()

        # Extract input features from request body
        prg = request_data.get("prg")
        pl = request_data.get("pl")
        pr = request_data.get("pr")
        sk = request_data.get("sk")
        ts = request_data.get("ts")
        m11 = request_data.get("m11")
        bd2 = request_data.get("bd2")
        age = request_data.get("age")

        # Prepare input features for prediction
        input_features = [prg, pl, pr, sk, ts, m11, bd2, age]

        # Make predictions using the loaded model
        prediction = xgb_model.predict([input_features])[0]

        # Determine the prediction outcome message
        if prediction == 0:
            prediction_message = "Patient has no sepsis"
        else:
            prediction_message = "Patient has sepsis"

        # Create a response dictionary
        response = {
            "request": {
                "prg": prg,
                "pl": pl,
                "pr": pr,
                "sk": sk,
                "ts": ts,
                "m11": m11,
                "bd2": bd2,
                "age": age
            },
            "prediction": {
                "class_0_probability": prediction[0],
                "class_1_probability": prediction[1],
                "prediction_message": prediction_message
            }
        }

        return JSONResponse(content=response, media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
