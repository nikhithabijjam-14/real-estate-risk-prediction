import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Page Config
st.set_page_config(
    page_title="Real Estate Risk Prediction",
    page_icon="🏠",
    layout="wide"
)

# Title
st.title("🏠 Real Estate Property Investment Risk Prediction")

st.write(
    "Machine Learning dashboard for predicting property investment risk."
)

# Generate Synthetic Dataset
np.random.seed(42)

n = 1000

property_price = np.random.randint(50000, 500000, n)

market_volatility = np.random.uniform(0.1, 1.0, n)

crime_rate = np.random.uniform(1, 10, n)

school_rating = np.random.uniform(1, 10, n)

infrastructure_score = np.random.uniform(1, 10, n)

loan_interest_rate = np.random.uniform(3, 12, n)

# Risk Score
risk_score = (
    market_volatility * 30 +
    crime_rate * 5 +
    loan_interest_rate * 2 -
    school_rating * 3 -
    infrastructure_score * 2
)

risk_level = []

for score in risk_score:

    if score < 25:
        risk_level.append("Low")

    elif score < 45:
        risk_level.append("Medium")

    else:
        risk_level.append("High")

# Create DataFrame
data = pd.DataFrame({

    "Property_Price": property_price,

    "Market_Volatility": market_volatility,

    "Crime_Rate": crime_rate,

    "School_Rating": school_rating,

    "Infrastructure_Score": infrastructure_score,

    "Loan_Interest_Rate": loan_interest_rate,

    "Risk_Level": risk_level
})

# Show Dataset
st.subheader("📊 Synthetic Dataset")

st.dataframe(data.head(20))

# Preprocessing
X = data.drop("Risk_Level", axis=1)

y = data["Risk_Level"]

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

st.subheader("✅ Model Accuracy")

st.success(f"Accuracy: {accuracy:.2f}")

# Sidebar
st.sidebar.header("🏘 Enter Property Details")

property_price_input = st.sidebar.slider(
    "Property Price",
    50000,
    500000,
    250000
)

market_volatility_input = st.sidebar.slider(
    "Market Volatility",
    0.1,
    1.0,
    0.5
)

crime_rate_input = st.sidebar.slider(
    "Crime Rate",
    1.0,
    10.0,
    5.0
)

school_rating_input = st.sidebar.slider(
    "School Rating",
    1.0,
    10.0,
    5.0
)

infrastructure_input = st.sidebar.slider(
    "Infrastructure Score",
    1.0,
    10.0,
    5.0
)

loan_interest_input = st.sidebar.slider(
    "Loan Interest Rate",
    3.0,
    12.0,
    6.0
)

# Predict Button
if st.sidebar.button("Predict Risk"):

    input_data = pd.DataFrame({

        "Property_Price": [property_price_input],

        "Market_Volatility": [market_volatility_input],

        "Crime_Rate": [crime_rate_input],

        "School_Rating": [school_rating_input],

        "Infrastructure_Score": [infrastructure_input],

        "Loan_Interest_Rate": [loan_interest_input]
    })

    prediction = model.predict(input_data)

    risk_label = encoder.inverse_transform(prediction)

    st.subheader("🏠 Predicted Investment Risk")

    if risk_label[0] == "High":

        st.error(f"Risk Level: {risk_label[0]}")

    elif risk_label[0] == "Medium":

        st.warning(f"Risk Level: {risk_label[0]}")

    else:

        st.success(f"Risk Level: {risk_label[0]}")

# Risk Distribution Graph
st.subheader("📈 Risk Distribution")

fig1, ax1 = plt.subplots()

sns.countplot(
    x="Risk_Level",
    data=data,
    ax=ax1
)

st.pyplot(fig1)

# Heatmap
st.subheader("🔥 Feature Correlation Heatmap")

fig2, ax2 = plt.subplots(figsize=(10, 6))

numeric_data = data.select_dtypes(
    include=["float64", "int64"]
)

sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax2
)

st.pyplot(fig2)

# Insights
st.subheader("💡 Investment Insights")

col1, col2 = st.columns(2)

with col1:

    st.error(
        '''
        High Risk Indicators

        • High market volatility
        • High crime rate
        • High loan interest rate
        • Poor infrastructure
        '''
    )

with col2:

    st.success(
        '''
        Low Risk Indicators

        • Good schools
        • Strong infrastructure
        • Stable market
        • Low crime rate
        '''
    )

# Footer
st.markdown("---")

st.markdown(
    "Developed using Streamlit and Machine Learning"
)