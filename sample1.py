import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters for the Gompertz model
a = 0.1  # Growth rate constant (per day)
K = 1000  # Carrying capacity (maximum tumor size, e.g., mm³ or cells)
x0 = 10   # Initial tumor size (e.g., mm³ or cells)

# Gompertz model differential equation
def gompertz_model(t, x):
    return a * x * np.log(K / x)

# Time range for simulation (e.g., 0 to 200 days)
t_span = (0, 200)  # Start and end time
t_eval = np.linspace(t_span[0], t_span[1], 500)  # Points to evaluate the solution

# Solve the differential equation
solution = solve_ivp(gompertz_model, t_span, [x0], t_eval=t_eval, method='RK45')

# Extract time and tumor size values
t_values = solution.t
x_values = solution.y[0]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t_values, x_values, label='Tumor Size (Adenocarcinoma)', color='blue')
plt.axhline(K, color='red', linestyle='--', label='Carrying Capacity (K)')
plt.title('Tumor Growth Simulation using Gompertz Model')
plt.xlabel('Time (days)')
plt.ylabel('Tumor Size (mm³)')
plt.legend()
plt.grid()
plt.show()
