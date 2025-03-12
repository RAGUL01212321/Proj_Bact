import numpy as np
import pandas as pd
import os

# Define the folder to store datasets
dataset_folder = "datasets"
os.makedirs(dataset_folder, exist_ok=True)

# Generate datasets for different nutrient levels
nutrient_levels = np.linspace(0.01, 0.1, 5)  # Example nutrient concentrations
num_samples = 1000  # Number of data points per dataset

for nutrient in nutrient_levels:
    time = np.linspace(0, 100, num_samples)
    growth = 10 / (1 + np.exp(-0.1 * (time - 50)))  # Logistic growth model
    noise = np.random.normal(scale=0.5, size=num_samples)  # Adding noise
    population = growth + noise
    
    df = pd.DataFrame({"Time": time, "Population": population})
    filename = f"{dataset_folder}/nutrient_{nutrient:.2f}.csv"
    df.to_csv(filename, index=False)
    
    print(f"Saved: {filename}")

print("Dataset generation complete!")
