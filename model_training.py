from sklearn.ensemble import RandomForestClassifier

import joblib


def train_model(X_train, y_train):

    # Create model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(
        model,
        "models/risk_model.pkl"
    )

    print("Model Trained Successfully")

    return model