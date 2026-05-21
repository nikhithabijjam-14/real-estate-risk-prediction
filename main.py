from src.preprocessing import load_and_preprocess_data

from src.model_training import train_model

from src.evaluation import evaluate_model

from src.visualization import generate_visualizations


# Load and preprocess data
X_train, X_test, y_train, y_test, encoder = load_and_preprocess_data()

# Train model
model = train_model(
    X_train,
    y_train
)

# Evaluate model
evaluate_model(
    model,
    X_test,
    y_test
)

# Generate graphs
generate_visualizations()