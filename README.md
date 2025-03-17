# Muscle Metrics: Estimating Trends in Myoelectric Activity Using Computer Vision and Machine Learning to Improve Weightlifting Form

This project uses computer vision and machine learning to analyze weightlifting form by predicting myoelectric activity (EMG signals) from video recordings. The system can provide feedback on proper form during dumbbell pressing movements, offering an accessible alternative to expensive motion capture systems or specialized EMG sensors.

## Project Overview

This system works by:
1. Collecting synchronized video footage and surface electromyography (sEMG) data
2. Processing video footage using OpenPose to extract body key points
3. Calculating unit vectors between key points to capture movement patterns
4. Using a neural network to predict EMG signals from pose data
5. Classifying the form as proper or poor based on sEMG signals

## Requirements

### Hardware
- Arduino Uno or similar microcontroller
- MyoWare Muscle 2.0 sensors (1+ sensors)
- Electrodes for sEMG sensors
- LED for synchronization
- Camera (webcam or smartphone)
- Computer with GPU (recommended for faster processing)

### Software
- Python 3.8+
- OpenPose

### Python Packages
- Create a python virtual environment then install requirements
```
pip install -r requirements.txt
```

## Necessary Files and Folders

```
├── data/                      # Directory for processed data
├── pose/                      # Directory for OpenPose output
├── semg/                      # Directory for EMG sensor data
├── saved_models/              # Directory for saving PyTorch models
├── classification.json        # Form classification labels (Key is session #, value is 1 for proper, 0 for poor {"0": "1", "1": "0"})
├── .env                       # Stores OPENPOSE_PATH variable which should be equal to absolute path for openPose.exe
```

## Setup Instructions

1. **Hardware Setup**
   - Connect the MyoWare sensors to the Arduino following proper electrode placement:
     - Deltoid
     - Pectoralis Major
     - Pectoralis Minor
     - Tricep (Long Head)
     - Tricep (Lateral Head)
   - Set up the LED for synchronization
   - Position the camera for a clear view of the subject

2. **Data Collection**
   - Create the necessary directories (`data`, `pose`, `semg`)
   - Record video footage of weightlifting exercises
   - Simultaneously collect sEMG data through the Arduino
   - Save the sEMG data from serial outputs as text files using PuTTy in the `semg` directory

3. **Install OpenPose**
   - Run the `setup.ps1` file to install openPose and initialize repository as needed
   - Run the openpose script to save the output JSON files in the 'pose' directory
      ```
     python openpose_script.py
     ```

4. **Process the Data**
   - Run the data processing script to create the necessary data files:
     ```
     python data_processing.py
     ```
   - This will generate:
     - `pose_data.npz`: Processed pose keypoints data
     - `semg_data.npz`: Processed EMG sensor data
     - `vector_data.npz`: Unit vectors between key points
     - `classification_data.npz`: Form classification data

5. **Train the Models**
   - Train the EMG prediction model:
     ```
     python emg_prediction_model.py
     ```
   - This will train and save the model as `saved_models/emg_estimator_model_10sec_1sensor.pth`
   - Train the form classification model:
     ```
     python form_classification_model.py
     ```
   - This will train and save the model as `saved_models/emg_estimator_model_10sec_1sensor.pth`


## Usage

1. **Visualize Sensor Data**
   - To visualize the collected EMG data:
     ```
     python data_graphing.py
     ```
   - This will create animated graphs showing the EMG sensor readings over time

2. **Make Predictions with Trained Models**
   - After training, you can use the models to:
     - Predict EMG values from new video footage
     - Classify form as proper or poor based on the predicted EMG values

3. **Interpreting Results**
   - The EMG prediction model shows trends in muscle activation during the exercise
   - The form classification model predicts whether the form is proper or poor