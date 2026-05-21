import pandas as pd
import numpy as np

# Random seed
np.random.seed(42)

# Number of records
n = 1000

# Generate synthetic data
property_price = np.random.randint(50000, 500000, n)

market_volatility = np.random.uniform(0.1, 1.0, n)

crime_rate = np.random.uniform(1, 10, n)

school_rating = np.random.uniform(1, 10, n)

infrastructure_score = np.random.uniform(1, 10, n)

loan_interest_rate = np.random.uniform(3, 12, n)

# Calculate risk score
risk_score = (
    market_volatility * 30 +
    crime_rate * 5 +
    loan_interest_rate * 2 -
    school_rating * 3 -
    infrastructure_score * 2
)

# Convert risk score into categories
risk_level = []

for score in risk_score:

    if score < 25:
        risk_level.append("Low")

    elif score < 45:
        risk_level.append("Medium")

    else:
        risk_level.append("High")

# Create dataframe
real_estate_data = pd.DataFrame({

    "Property_Price": property_price,

    "Market_Volatility": market_volatility,

    "Crime_Rate": crime_rate,

    "School_Rating": school_rating,

    "Infrastructure_Score": infrastructure_score,

    "Loan_Interest_Rate": loan_interest_rate,

    "Risk_Level": risk_level
})

# Save dataset
real_estate_data.to_csv(
    "data/synthetic_real_estate_data.csv",
    index=False
)

print("Dataset Generated Successfully")

print(real_estate_data.head())