from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load the trained model
model_path = "models/xgboost_smote_tuned.pkl"  # Ensure the model is in the correct path
try:
    model = joblib.load(model_path)
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Model file '{model_path}' not found. Make sure it is in the correct path.")
    model = None

# Define the expected feature names
EXPECTED_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

# Define default values for missing features
DEFAULT_VALUES = {
    "gender": 0, "SeniorCitizen": 0, "Partner": 0, "Dependents": 0,
    "tenure": 0.0, "PhoneService": 0, "MultipleLines": 0, "InternetService": 0,
    "OnlineSecurity": 0, "OnlineBackup": 0, "DeviceProtection": 0, "TechSupport": 0,
    "StreamingTV": 0, "StreamingMovies": 0, "Contract": 0, "PaperlessBilling": 0,
    "PaymentMethod": 0, "MonthlyCharges": 0.0, "TotalCharges": 0.0
}

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not found. Ensure the model file exists and is correctly loaded."}), 500

    try:
        # Get JSON data from the request
        data = request.get_json()
        print("📥 Received Data:", data)  # Debugging log

        # Fill missing fields with default values
        for field in EXPECTED_FEATURES:
            if field not in data:
                print(f"⚠️ Warning: Missing field '{field}', using default value: {DEFAULT_VALUES[field]}")
                data[field] = DEFAULT_VALUES[field]

        # Convert input data into a Pandas DataFrame
        df = pd.DataFrame([data])

        # Ensure correct data types
        df = df.astype({
            "gender": int, "SeniorCitizen": int, "Partner": int, "Dependents": int,
            "tenure": float, "PhoneService": int, "MultipleLines": int, "InternetService": int,
            "OnlineSecurity": int, "OnlineBackup": int, "DeviceProtection": int, "TechSupport": int,
            "StreamingTV": int, "StreamingMovies": int, "Contract": int, "PaperlessBilling": int,
            "PaymentMethod": int, "MonthlyCharges": float, "TotalCharges": float
        })

        # Ensure feature order matches the model
        df = df[EXPECTED_FEATURES]

        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        print(f"🔮 Prediction: {prediction}, Probability: {probability:.4f}")  # Debugging log

        return jsonify({
            "prediction": int(prediction),
            "probability": round(float(probability), 4)
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")  # Log the error
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
