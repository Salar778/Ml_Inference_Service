import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field 
from typing import List

Model_Path = os.path.join("model", "iris_model.joblib")
app = FastAPI(title="ML Inference API", version="1.0.0" )

artifact = None

class PredictRequest(BaseModel):
    features: List[float] = Field (...,min_items=4, max_items=4, example=[5.1, 3.5, 1.4, 0.2])

class PredictResponse(BaseModel):
    predicted_class: str
    predicted_class_id: int

@app.on_event("startup")
def loadmodel():
    global artifact
    if not os.path.exists(Model_Path):
        raise RuntimeError(f"Model not found at {Model_Path}. Train the model first.")  
    artifact = joblib.load(Model_Path)

@app.get("/health")
def health():
    return{"status": "ok"}

@app.get("/metadata")
def metadata():
    if artifact is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return{
        "feature_names": artifact["feature_names"],
        "target_names": artifact["target_names"],
        "offline_test_accuracy": artifact["accuracy"],
        "model_path": Model_Path        
        }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if artifact is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    model= artifact["model"]
    target_names= artifact["target_names"]
    
    try:
        pred_id = int(model.predict([req.features])[0])
        pred_name = target_names[pred_id]
        return PredictResponse(predicted_class=pred_name, predicted_class_id=pred_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")        

    
