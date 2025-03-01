import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def animated_graph(session_file_name: str, graph_folder: str = "graphs") -> None:
    """Animates the sensor data for all five muscles for one session file
    Args:
        session_file_name (str): path to the session file
        graph_folder (str, optional): folder to save the animated graph. Defaults to "graphs".
    """
    # Read the session file
    with open(session_file_name, 'r') as file:
        sessionData = file.readlines()
    # Filter out non-numeric lines
    session_clean_data = []
    for line in sessionData:
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
    session_dataframe = pd.DataFrame(session_sensor_data)
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
                session_dataframe["Frame Number"][:frame],
                session_dataframe[sensor][:frame]
            )
        return lines
    # Number of frames and interval for animation
    fps = 3000000
    interval = 1000 // fps  # Interval in milliseconds
    frames = len(session_dataframe)
    # Create the animation
    ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
    # Show the plot
    plt.show()

if __name__ == "__main__":
    semg_folder = 'semg'
    for semg_file in os.listdir(semg_folder):
        animated_graph(f'{semg_folder}/{semg_file}')