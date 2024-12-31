import numpy as np
import matplotlib.pyplot as plt

# Simulating bacterial growth (without drug)
time_steps = 100  # Number of time steps
growth_rate = 0.1  # Growth rate (per hour)
initial_population = 1  # Initial bacterial population

# Generate time series data (logistic growth model without drug)
time = np.arange(0, time_steps)
bacterial_population = initial_population * np.exp(growth_rate * time)

# Add some noise to simulate realistic biological variations
noise = np.random.normal(0, 0.05, bacterial_population.shape)
bacterial_population_noisy = bacterial_population + noise

# Plot the generated data (bacterial growth)
plt.plot(time, bacterial_population_noisy, label='Bacterial Growth (No Drug)')
plt.xlabel('Time (hours)')
plt.ylabel('Bacterial Population')
plt.legend()
plt.show()
