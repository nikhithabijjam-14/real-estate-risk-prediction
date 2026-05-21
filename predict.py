import joblib

import pandas as pd


# Load trained model
model = joblib.load(
    "models/risk_model.pkl"
)

# New property data
new_property = pd.DataFrame({

    "Property_Price": [250000],

    "Market_Volatility": [0.8],

    "Crime_Rate": [8],

    "School_Rating": [4],

    "Infrastructure_Score": [5],

    "Loan_Interest_Rate": [10]
})

# Predict
prediction = model.predict(new_property)

# Risk mapping
risk_mapping = {

    0: "High",

    1: "Low",

    2: "Medium"
}

print("Predicted Risk Level:")

print(
    risk_mapping[prediction[0]]
)