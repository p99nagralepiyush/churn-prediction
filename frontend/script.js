document.getElementById('prediction-form').addEventListener('submit', function (e) {
    e.preventDefault();

    // Get form data
    const formData = {
        gender: document.getElementById('gender').value,
        SeniorCitizen: document.getElementById('SeniorCitizen').value,
        Partner: document.getElementById('Partner').value,
        Dependents: document.getElementById('Dependents').value,
        tenure: document.getElementById('tenure').value,
        MonthlyCharges: document.getElementById('MonthlyCharges').value,
        TotalCharges: document.getElementById('TotalCharges').value
    };

    console.log('Form Data:', formData); // Debugging step to check what is being sent

    // Send data to the API
    fetch('https://churn-prediction-ekdt.onrender.com/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('API Response:', data); // Debugging step to check API response

        if (!data || typeof data.probability === 'undefined') {
            throw new Error('Invalid API response: Missing "probability" field');
        }

        document.getElementById('prediction').textContent = data.prediction === 1 ? 'Churn' : 'No Churn';
        document.getElementById('probability').textContent = data.probability.toFixed(4);
        document.getElementById('result').classList.remove('hidden');
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
    });
});
