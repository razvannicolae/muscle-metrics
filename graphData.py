import pandas as pd
import matplotlib.pyplot as plt

# Load the newly uploaded file's data again
session_file_path = 'C:\\Users\\Shreyas\\Downloads\\session0.txt'
with open(session_file_path, 'r') as file:
    session_data = file.readlines()

# Filter out non-numeric lines
session_clean_data = []
for line in session_data:
    line = line.strip()
    if line.isdigit():
        session_clean_data.append(line)

# Initialize a new dictionary for the sensor data
session_sensor_data = {
    "Deltoid Sensor": [],
    "Pec Major Sensor": [],
    "Pec Minor Sensor": [],
    "Tricep (Long Head) Sensor": [],
    "Tricep (Lateral Head) Sensor": [],
    "Time Reading": [],
    "Frame Number": []
}

# Parse the cleaned data
for i in range(0, len(session_clean_data), 7):
    try:
        session_sensor_data["Deltoid Sensor"].append(int(session_clean_data[i]))
        session_sensor_data["Pec Major Sensor"].append(int(session_clean_data[i+1]))
        session_sensor_data["Pec Minor Sensor"].append(int(session_clean_data[i+2]))
        session_sensor_data["Tricep (Long Head) Sensor"].append(int(session_clean_data[i+3]))
        session_sensor_data["Tricep (Lateral Head) Sensor"].append(int(session_clean_data[i+4]))
        session_sensor_data["Time Reading"].append(int(session_clean_data[i+5]))
        session_sensor_data["Frame Number"].append(int(session_clean_data[i+6]))
    except IndexError:
        # Handle incomplete data
        break

# Align the data lengths by trimming to the shortest length
min_length_session = min(len(session_sensor_data[key]) for key in session_sensor_data)
for key in session_sensor_data:
    session_sensor_data[key] = session_sensor_data[key][:min_length_session]

# Create a DataFrame from the session data
session_df = pd.DataFrame(session_sensor_data)

# Plotting each sensor's values against the frame number for the new session file
fig, axs = plt.subplots(5, 1, figsize=(10, 20), constrained_layout=True)

for i, sensor in enumerate(["Deltoid Sensor", "Pec Major Sensor", "Pec Minor Sensor", "Tricep (Long Head) Sensor", "Tricep (Lateral Head) Sensor"]):
    axs[i].plot(session_df["Frame Number"], session_df[sensor], label=sensor, marker='o')
    axs[i].set_xlabel("Frame Number")
    axs[i].set_ylabel(sensor)
    axs[i].set_title(f"{sensor} vs Frame Number (Session File)")
    axs[i].grid(True)
    axs[i].legend()

plt.show()

