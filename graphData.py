import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load the newly uploaded file's data again
sessionFilePath = 'semg\\session0.txt'
with open(sessionFilePath, 'r') as file:
    sessionData = file.readlines()

# Filter out non-numeric lines
sessionCleanData = []
for line in sessionData:
    line = line.strip()
    if line.isdigit():
        sessionCleanData.append(line)

# Initialize a new dictionary for the sensor data
sessionSensorData = {
    "Deltoid Sensor": [],
    "Pec Major Sensor": [],
    "Pec Minor Sensor": [],
    "Tricep (Long Head) Sensor": [],
    "Tricep (Lateral Head) Sensor": [],
    "Time Reading": [],
    "Frame Number": []
}

# Parse the cleaned data
for i in range(0, len(sessionCleanData), 7):
    try:
        sessionSensorData["Deltoid Sensor"].append(int(sessionCleanData[i]))
        sessionSensorData["Pec Major Sensor"].append(int(sessionCleanData[i+1]))
        sessionSensorData["Pec Minor Sensor"].append(int(sessionCleanData[i+2]))
        sessionSensorData["Tricep (Long Head) Sensor"].append(int(sessionCleanData[i+3]))
        sessionSensorData["Tricep (Lateral Head) Sensor"].append(int(sessionCleanData[i+4]))
        sessionSensorData["Time Reading"].append(int(sessionCleanData[i+5]))
        sessionSensorData["Frame Number"].append(int(sessionCleanData[i+6]))
    except IndexError:
        # Handle incomplete data
        break

# Align the data lengths by trimming to the shortest length
minLengthSession = min(len(sessionSensorData[key]) for key in sessionSensorData)
for key in sessionSensorData:
    sessionSensorData[key] = sessionSensorData[key][:minLengthSession]

# Create a DataFrame from the session data
sessionDf = pd.DataFrame(sessionSensorData)

# Plotting each sensor's values against the frame number for the new session file
fig, axs = plt.subplots(5, 1, figsize=(10, 20), constrained_layout=True)

lines = []
for i, sensor in enumerate(["Deltoid Sensor", "Pec Major Sensor", "Pec Minor Sensor", "Tricep (Long Head) Sensor", "Tricep (Lateral Head) Sensor"]):
    line, = axs[i].plot([], [], label=sensor, marker='o', markersize = 0)
    axs[i].set_xlim(0, 300)
    axs[i].set_ylim(0, 1000)
    axs[i].set_xlabel("Frame Number")
    axs[i].set_ylabel(sensor)
    axs[i].set_title(f"{sensor} vs Frame Number (Session File)")
    axs[i].grid(True)
    axs[i].legend()
    lines.append(line)

# Update function for animation
def update(frame):
    for i, sensor in enumerate(["Deltoid Sensor", "Pec Major Sensor", "Pec Minor Sensor", "Tricep (Long Head) Sensor", "Tricep (Lateral Head) Sensor"]):
        lines[i].set_data(
            sessionDf["Frame Number"][:frame],
            sessionDf[sensor][:frame]
        )
    return lines

# Number of frames and interval for animation
fps = 3
interval = 1000 // fps  # Interval in milliseconds
frames = len(sessionDf)

# Create the animation
ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)

plt.show()