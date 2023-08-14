from fastapi import FastAPI,Query
from pydantic import BaseModel

# instantiating the app

app= FastAPI()

class patient_item(BaseModel):
    prg: float = Query(..., description="Plasma glucose"),
    pl: float = Query(..., description="Blood Work Result-1 (mu U/ml)"),
    pr: float = Query(..., description="Blood Pressure (mm Hg)"),
    sk: float = Query(..., description="Blood Work Result-2 (mm)"),
    ts: float = Query(..., description="Blood Work Result-3 (mu U/ml)"),
    m11: float = Query(..., description="Body mass index (weight in kg/(height in m)^2"),
    bd2: float = Query(..., description="Blood Work Result-4 (mu U/ml)"),
    age: int = Query(..., description="Patient's age (years)")   

model= joblib.load('xgb_model.joblib')

@app.get('/')
async def patient_endpoint(item:patient_item):
    df= pd.DataFrame([item.dict().values], columns= item.dict().keys())
    yhat= model.predict(df)
    return{'prediction':int(yhat)} 
    
    
