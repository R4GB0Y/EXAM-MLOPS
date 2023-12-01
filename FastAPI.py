import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import numpy as np
import onnxruntime as ort

# Initialize FastAPI
app = FastAPI()

# Load the ONNX model
onnx_model_path = 'random_forest_regression.onnx'
session = ort.InferenceSession(onnx_model_path)

# Load the scaler
scaler_path = 'scaler.pkl'
scaler = joblib.load(scaler_path)

class PredictionRequest(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Preprocess the input data
        input_data = np.array([
            [
                request.bedrooms, request.bathrooms, request.sqft_living,
                request.sqft_lot, request.floors, request.waterfront,
                request.view, request.condition, request.grade,
                request.sqft_above, request.sqft_basement, request.yr_built,
                request.yr_renovated, request.zipcode, request.lat,
                request.long, request.sqft_living15, request.sqft_lot15
            ]
        ])

        scaled_data = scaler.transform(input_data)

        # Make a prediction using the ONNX model
        onnx_inputs = {session.get_inputs()[0].name: scaled_data.astype(np.float32)}
        onnx_output = session.run(None, onnx_inputs)

        # Extract the prediction
        prediction = onnx_output[0][0]

        return {"prediction": prediction}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Prediction failed: {str(e)}"})

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8080, reload=True)
