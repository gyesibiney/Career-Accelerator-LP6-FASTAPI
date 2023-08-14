from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib

app = FastAPI()
@app.post("/inference")
templates = Jinja2Templates(directory="templates")

# Load the pickled XGBoost model
xgb_model = joblib.load("xgb_model.joblib")

@app.get("/")
async def read_root():
    return {"message": "Welcome to my Sepsis-Prediction-APP-using-FastAPI"}


@app.get("/predict/")
async def predict_diabetes(
    request: Request,
    prg: float = Query(..., description="Plasma glucose"),
    pl: float = Query(..., description="Blood Work Result-1 (mu U/ml)"),
    pr: float = Query(..., description="Blood Pressure (mm Hg)"),
    sk: float = Query(..., description="Blood Work Result-2 (mm)"),
    ts: float = Query(..., description="Blood Work Result-3 (mu U/ml)"),
    m11: float = Query(..., description="Body mass index (weight in kg/(height in m)^2"),
    bd2: float = Query(..., description="Blood Work Result-4 (mu U/ml)"),
    age: int = Query(..., description="Patient's age (years)")
):
    try:
        # Prepare input features for prediction
        input_features = [prg, pl, pr, sk, ts, m11, bd2, age]

        # Make predictions using the loaded model
        prediction = xgb_model.predict_proba([input_features])[0]

        # Create a JSON response
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
                "class_1_probability": prediction[1]
            }
        }

        return templates.TemplateResponse(
            "display_params.html",
            {
                "request": request,
                "input_features": response["request"],
                "prediction": response["prediction"]
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred while processing the request.")