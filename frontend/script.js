document.addEventListener("DOMContentLoaded", function () {
    const predictionForm = document.getElementById("prediction-form");
    const resultDiv = document.getElementById("result");
    const predictionText = document.getElementById("prediction");
    const probabilityText = document.getElementById("probability");

    if (!predictionForm || !resultDiv || !predictionText || !probabilityText) {
        console.error("Error: Some required elements are missing from the HTML.");
        return;
    }

    predictionForm.addEventListener("submit", function (e) {
        e.preventDefault();

        // Get form data
        const formData = {
            gender: parseInt(document.getElementById("gender").value),
            SeniorCitizen: parseInt(document.getElementById("SeniorCitizen").value),
            Partner: parseInt(document.getElementById("Partner").value),
            Dependents: parseInt(document.getElementById("Dependents").value),
            tenure: parseFloat(document.getElementById("tenure").value),
            PhoneService: parseInt(document.getElementById("PhoneService").value),
            MultipleLines: parseInt(document.getElementById("MultipleLines").value),
            InternetService: parseInt(document.getElementById("InternetService").value),
            OnlineSecurity: parseInt(document.getElementById("OnlineSecurity").value),
            OnlineBackup: parseInt(document.getElementById("OnlineBackup").value),
            DeviceProtection: parseInt(document.getElementById("DeviceProtection").value),
            TechSupport: parseInt(document.getElementById("TechSupport").value),
            StreamingTV: parseInt(document.getElementById("StreamingTV").value),
            StreamingMovies: parseInt(document.getElementById("StreamingMovies").value),
            Contract: parseInt(document.getElementById("Contract").value),
            PaperlessBilling: parseInt(document.getElementById("PaperlessBilling").value),
            PaymentMethod: parseInt(document.getElementById("PaymentMethod").value),
            MonthlyCharges: parseFloat(document.getElementById("MonthlyCharges").value),
            TotalCharges: parseFloat(document.getElementById("TotalCharges").value)
        };        

        console.log("Form Data Sent to API:", formData); // Debugging log

        // Send data to the API
        fetch("https://churn-prediction-ekdt.onrender.com/predict", {  // Replace with actual API URL
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(formData)
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { 
                    throw new Error(`HTTP ${response.status}: ${err.error || "Unknown error"}`); 
                });
            }
            return response.json();
        })
        .then(data => {
            console.log("API Response:", data); // Debugging log

            if (!data || typeof data.probability === "undefined") {
                throw new Error("Invalid API response: Missing 'probability' field");
            }

            // Display the result
            predictionText.textContent = data.prediction === 1 ? "Churn" : "No Churn";
            probabilityText.textContent = (data.probability * 100).toFixed(2) + "%";
            resultDiv.classList.remove("hidden");
        })
        .catch(error => {
            console.error("Error:", error);
        });
    });
});
