import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns


def generate_visualizations():

    # Load dataset
    data = pd.read_csv(
        "data/synthetic_real_estate_data.csv"
    )

    # Risk distribution graph
    plt.figure(figsize=(6, 4))

    sns.countplot(
        x="Risk_Level",
        data=data
    )

    plt.title("Risk Level Distribution")

    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(8, 6))

    numeric_data = data.select_dtypes(
        include=["float64", "int64"]
    )

    sns.heatmap(
        numeric_data.corr(),
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Feature Correlation Heatmap")

    plt.show()