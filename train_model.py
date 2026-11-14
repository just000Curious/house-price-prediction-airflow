import os
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_and_save_model():
    print("1. Fetching housing dataset...")
    # Load dataset as a pandas DataFrame
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    # Save a copy of the raw data to our local data folder
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/house_data.csv", index=False)
    print(f" Raw dataset saved to data/house_data.csv | Shape: {df.shape}")

    # Separate features (X) from the target price we want to predict (y)
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    print("2. Training the Random Forest model...")
    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    print(" Model training complete!")

    # Save the trained model to disk as a pickle file
    os.makedirs("models", exist_ok=True)
    model_path = "models/house_model.pkl"
    joblib.dump(model, model_path)
    print(f" Trained model saved successfully to {model_path}!")

if __name__ == "__main__":
    train_and_save_model()