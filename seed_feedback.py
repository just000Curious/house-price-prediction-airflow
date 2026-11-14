import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)
np.random.seed(42)

# Generate 50 realistic dummy feedback rows matching California Housing schema
n_samples = 50
dummy_data = pd.DataFrame({
    "MedInc": np.random.uniform(1.5, 8.5, n_samples).round(4),
    "HouseAge": np.random.randint(5, 50, n_samples).astype(float),
    "AveRooms": np.random.uniform(3.5, 7.5, n_samples).round(4),
    "AveBedrms": np.random.uniform(0.9, 1.4, n_samples).round(4),
    "Population": np.random.randint(300, 3500, n_samples).astype(float),
    "AveOccup": np.random.uniform(1.8, 4.2, n_samples).round(4),
    "Latitude": np.random.uniform(33.5, 38.0, n_samples).round(2),
    "Longitude": np.random.uniform(-122.4, -117.0, n_samples).round(2),
})

# Realistic target correlation with MedInc + slight noise
dummy_data["MedHouseVal"] = (dummy_data["MedInc"] * 0.45 + np.random.normal(0.3, 0.25, n_samples)).clip(0.5, 5.0).round(4)

feedback_path = "data/feedback_submissions.csv"
if os.path.exists(feedback_path):
    dummy_data.to_csv(feedback_path, mode="a", header=False, index=False)
else:
    dummy_data.to_csv(feedback_path, index=False)

print(f"Added {n_samples} dummy rows to {feedback_path}!")

