import serial
import matplotlib.pyplot as plt
from collections import deque
import time

# Set up the serial connection (adjust COM port and baud rate if needed)
ser = serial.Serial('COM5', 9600)  # Change 'COM5' to your Arduino port
time.sleep(2)  # Wait for the connection to establish

# Data storage
max_len = 50  # Number of points to display
voltage_data = deque(maxlen=max_len)
growth_rate_data = deque(maxlen=max_len)
time_data = deque(maxlen=max_len)

# Turn on interactive mode
plt.ion()
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle("Bacterial Growth Monitoring", fontsize=16)

# Function to update the plot
def update_plot():
    ax[0].clear()
    ax[1].clear()
    
    # Plot bacterial growth signal
    ax[0].plot(time_data, voltage_data, color='blue', label='Bacterial Growth Signal')
    ax[0].set_ylabel('Voltage (V)')
    ax[0].set_title('Bacterial Growth Signal over Time')
    ax[0].legend()
    ax[0].grid(True)
    
    # Display latest value
    if time_data:
        ax[0].text(time_data[-1], voltage_data[-1], f'{voltage_data[-1]:.2f} V', 
                   color='blue', fontsize=10, ha='right')

    # Plot growth rate
    ax[1].plot(time_data, growth_rate_data, color='red', label='Growth Rate')
    ax[1].set_ylabel('Growth Rate (V/s)')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_title('Growth Rate over Time')
    ax[1].legend()
    ax[1].grid(True)
    
    # Display latest value
    if time_data:
        ax[1].text(time_data[-1], growth_rate_data[-1], f'{growth_rate_data[-1]:.4f} V/s', 
                   color='red', fontsize=10, ha='right')
    
    plt.pause(0.1)

# Start time
start_time = time.time()

# Read and process data
while True:
    if ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if "Bacterial Growth Signal" in line:
                parts = line.split(',')
                voltage = float(parts[0].split(': ')[1].replace('V', ''))
                growth_rate = float(parts[1].split(': ')[1].replace('V/s', ''))
                
                # Store data
                current_time = round(time.time() - start_time, 2)
                voltage_data.append(voltage)
                growth_rate_data.append(growth_rate)
                time_data.append(current_time)

                # Print to terminal
                print(f'Time: {current_time}s, Voltage: {voltage:.2f}V, Growth Rate: {growth_rate:.4f}V/s')

                # Update the plot
                update_plot()

        except Exception as e:
            print(f"Error: {e}")

