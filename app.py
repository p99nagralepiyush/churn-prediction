from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow all origins

# Load the trained model
model_path = "models/xgboost_smote_tuned.pkl"  # Ensure the model is in the correct path
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    print(f"Error: Model file '{model_path}' not found. Make sure it is in the correct path.")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received Data:", data)  # Debugging: Check if request data is correct

        # Validate required fields
        required_fields = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "MonthlyCharges", "TotalCharges"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Convert data to DataFrame for model prediction
        df = pd.DataFrame([data])

        # Ensure numerical columns are correctly formatted
        df = df.astype({
            "gender": int,
            "SeniorCitizen": int,
            "Partner": int,
            "Dependents": int,
            "tenure": float,
            "MonthlyCharges": float,
            "TotalCharges": float
        })

        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return jsonify({"prediction": int(prediction), "probability": float(probability)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
