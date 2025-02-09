document.getElementById("prediction-form").addEventListener("submit", function (e) {
    e.preventDefault();

    // Get form data
    const formData = {
        gender: parseInt(document.getElementById("gender").value),
        SeniorCitizen: parseInt(document.getElementById("SeniorCitizen").value),
        Partner: parseInt(document.getElementById("Partner").value),
        Dependents: parseInt(document.getElementById("Dependents").value),
        tenure: parseFloat(document.getElementById("tenure").value),
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

        // Update UI with prediction results
        document.getElementById("prediction").textContent = data.prediction === 1 ? "Churn" : "No Churn";
        document.getElementById("probability").textContent = data.probability.toFixed(4);
        document.getElementById("result").classList.remove("hidden");
    })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById("result").innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
        document.getElementById("result").classList.remove("hidden");
    });
});
