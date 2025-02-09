from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

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

        # Example prediction output
        prediction = 1  # 1 = Churn, 0 = No Churn
        probability = 0.85  # Example probability

        return jsonify({"prediction": prediction, "probability": probability})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
