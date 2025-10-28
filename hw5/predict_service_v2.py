import pickle
from fastapi import FastAPI
import uvicorn
from typing import Dict, Any

app = FastAPI(title="lead-scoring-prediction")

with open('pipeline_v2.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

@app.post("/predict")
def predict(customer: Dict[str, Any]):
    return pipeline.predict_proba(customer)[0, 1]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
