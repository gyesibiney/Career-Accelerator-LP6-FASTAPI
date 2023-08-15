from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
import joblib

app = FastAPI()

# Load the XGBoost model
model = joblib.load("Ada.joblib")

@app.get("/")
async def read_root():
    return {"message": "Welcome to my Sepsis-Prediction-APP-using-FastAPI"}

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
    try:
        # Prepare input features for prediction
        input_features = [prg, pl, pr, sk, ts, m11, bd2, age]

        # Convert float values to strings
        input_features = [str(value) for value in input_features]

        # Make predictions using the loaded model
        prediction = model.predict([input_features])[0]

        # Calculate class probabilities
        class_1_probability = prediction
        class_0_probability = 1 - class_1_probability
       
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
                "class_0_probability": class_0_probability,
                "class_1_probability": class_1_probability,
                "prediction_message": prediction_message
            }
        }

        return JSONResponse(content=response, media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# from fastapi import FastAPI,Query
# import joblib
# import uvicorn
# from pydantic import BaseModel
# from sklearn.pipeline import Pipeline, make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer, make_column_selector as selector
# import pandas as pd
# import numpy as np



# app = FastAPI(title ='Sepsis Prediction APP', version = 1.0, description = 'Classification Machine Learning Prediction')

# class model_input(BaseModel):
#     prg: float = Query(..., description="Plasma glucose"),
#     pl: float = Query(..., description="Blood Work Result-1 (mu U/ml)"),
#     pr: float = Query(..., description="Blood Pressure (mm Hg)"),
#     sk: float = Query(..., description="Blood Work Result-2 (mm)"),
#     ts: float = Query(..., description="Blood Work Result-3 (mu U/ml)"),
#     m11: float = Query(..., description="Body mass index (weight in kg/(height in m)^2"),
#     bd2: float = Query(..., description="Blood Work Result-4 (mu U/ml)"),
#     age: int = Query(..., description="Patient's age (years)")

  

# # load the model
# model = joblib.load("Ada.joblib")


  
# @app.post("/Sepsis_App_prediction")
# async def predicts(input:model_input):
#     # Numeric Features
#     num_attr = [["Plasma_glucose", "Blood_Work_Result-1", "Blood_Pressure", "Blood_Work_Result-2", "Blood_Work_Result-3", 
#                    "Body_mass_index", "Blood_Work_Result-4", "patients_age"]]
    
#     num_pipeline = Pipeline([('imputer', SimpleImputer()),('scaler', StandardScaler())])
    

#     full_pipeline =ColumnTransformer([('num_pipe',num_pipeline,num_attr)])
    
#     #print(model_input)
#     df = pd.DataFrame([input])
#     final_input = np.array(full_pipeline.fit_transform(df), dtype = np.str)

#     prediction = model.predict(np.array([[final_input]]).reshape(1, 1))
#     return prediction

# # if __name__ == '__main__':
# #     uvicorn.run("Main:app", reload = True)
