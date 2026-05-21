import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data():

    # Load dataset
    data = pd.read_csv(
        "data/synthetic_real_estate_data.csv"
    )

    # Features
    X = data.drop("Risk_Level", axis=1)

    # Target
    y = data["Risk_Level"]

    # Encode labels
    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test, encoder