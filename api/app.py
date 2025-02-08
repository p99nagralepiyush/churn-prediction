from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

# Load the trained model
model = joblib.load('../models/xgboost_smote_tuned.pkl')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow all origins

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    input_data = request.json

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

    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)