import pandas as pd
import matplotlib.pyplot as plt

# Path to your dataset
file_path = r"C:\Users\uragu\OneDrive\Desktop\Analog Sample\datasets\nutrient_0.01.csv"

# Load the dataset
df = pd.read_csv(file_path)

# Plot the data
plt.figure(figsize=(8, 5))
plt.plot(df["Time"], df["Population"], marker='o', linestyle='-', color='b', label="Growth Curve")

# Labels and title
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Bacterial Growth Over Time (Nutrient 0.01%)")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
