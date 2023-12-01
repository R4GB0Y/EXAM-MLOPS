from flask import Flask, request, jsonify, render_template
import numpy as np
import onnxruntime as ort
import joblib

app = Flask(__name__)

# Load the ONNX model
onnx_model_path = 'random_forest_regression.onnx'
session = ort.InferenceSession(onnx_model_path)

# Load the scaler
scaler_path = 'scaler.pkl'
scaler = joblib.load(scaler_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.get_json()
        bedrooms = data['bedrooms']
        bathrooms = data['bathrooms']
        sqft_living = data['sqft_living']
        sqft_lot = data['sqft_lot']
        floors = data['floors']
        waterfront = data['waterfront']
        view = data['view']
        condition = data['condition']
        grade = data['grade']
        sqft_above = data['sqft_above']
        sqft_basement = data['sqft_basement']
        yr_built = data['yr_built']
        yr_renovated = data['yr_renovated']
        zipcode = data['zipcode']
        lat = data['lat']
        long = data['long']
        sqft_living15 = data['sqft_living15']
        sqft_lot15 = data['sqft_lot15']

        # Preprocess the input data
        input_data = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view,
                                condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated,
                                zipcode, lat, long, sqft_living15, sqft_lot15]])
        scaled_data = scaler.transform(input_data)

        # Make a prediction using the ONNX model
        onnx_inputs = {session.get_inputs()[0].name: scaled_data.astype(np.float32)}
        onnx_output = session.run(None, onnx_inputs)

        # Extract the prediction
        prediction = onnx_output[0]

        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
