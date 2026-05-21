from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix


def evaluate_model(model, X_test, y_test):

    # Predict
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\nModel Accuracy:")
    print(accuracy)

    # Classification Report
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions
        )
    )

    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )