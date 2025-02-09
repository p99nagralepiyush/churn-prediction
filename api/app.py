from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow all origins

# Load the trained model
model_path = '../models/xgboost_smote_tuned.pkl'  # Ensure the model is in the same directory
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    print(f"Error: Model file '{model_path}' not found. Make sure it is in the correct path.")
    model = None

# Define a test route to check if the API is running
@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Customer Churn Prediction API is running'}), 200

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Get the input data from the request
        input_data = request.get_json()

        # Convert input data into a DataFrame
        input_df = pd.DataFrame([input_data])

        # Make predictions
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)[:, 1]

        # Prepare the response
        response = {
            'prediction': int(prediction[0]),
            'probability': float(prediction_proba[0])
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use environment variable or default to 5000
    app.run(host='0.0.0.0', port=port, debug=True)