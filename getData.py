import numpy as np
import json
import os

"""
Data Structure:
openpose/
  ├── data/                                # All .npz files
  │   ├── combineddata.npz                 # Combined pose keypoint and sEMG data
  │   │   ├── session1                     # Data for first session
  │   │   └── ...              
  │   ├── framedata.npz                    # Pose keypoint data
  │   │   ├── session1                     # Data for first session
  │   │   └── ...                
  │   └── semgdata.npz                     # sEMG data
  │       ├── session1                     # Data for first session
  │       └── ...                
  │
  ├── pose/                                # Individual frame data
  │   ├── frame1/                          # Data for first session
  │   │   ├── 000000000000_keypoints.json  # Keypoints for frame 0
  │   │   └── ...                          
  │   └── ...                              
  │       └── ...
  │
  └── semg/                                # sEMG data files
      ├── session1.txt                     # sEMG data for first session
      └── ...                   

Data Specifications:
- JSON files follow OpenPose output format for keypoint data
- .npz files have a U32 dtype, will be converted when inputted into PyTorch
"""

def createFrameData():
    '''Given openPose JSON output, creates frameData.npz file with pose tracking data'''
    # Main directory for openPose JSON Outputs
    video_folder = 'pose'
    # Dictionary that stores all np arrays for each session
    frame_data_dict = {}
    # List of subfolders from each session with individual frame JSON keypoints data
    subfolders = os.listdir(video_folder)
    # Iterate through all subfolders
    for subfolder in subfolders:
        # Get all the JSON frame files from each session
        files = os.listdir(f"{video_folder}/{subfolder}/")
        sessionNum = subfolder[7:]
        # Create empty numpy array to append data to
        frameData = np.empty(shape=[0, 77])
        # Iterate through all files in respective subfolder
        for file in files:
            file_path = f"{video_folder}/{subfolder}/{file}"
            # Open JSON file to extract data
            with open(file_path, 'r') as json_file:
                json_data = json.load(json_file)
                frameNum = file[0:file.index("_")]
                # Append Frame 
                frameData = np.append(frameData, [[sessionNum, str(int(frameNum))] + json_data["people"][0]["pose_keypoints_2d"]], axis = 0)
        # Once all of the frames have been added to frameData add the numpy array for the session to the dict
        frame_data_dict[sessionNum] = frameData
    # Save all arrays for each recording session into a zipped numpy file
    np.savez('data/frameData.npz', **frame_data_dict)

def createSemgData():
    '''Given txt file from arduino serial output, creates semgData.npz file with semg data'''

def createFrameData():
    '''Given semgData.npz and frameData.npz file, creates combinedData.npz file with synched data'''

# -------------------------------- TESTING --------------------------------

# Load data from npz file add sEMG readings and save again
loaded_data = np.load('data/frameData.npz')
combined_data_dict = {}
for session in loaded_data.files:
    fakeSemgData = np.random.rand(loaded_data[session].shape[0], 5)
    combined_data_dict[session] = np.hstack((loaded_data[session], fakeSemgData))
np.savez('data/combinedData.npz', **combined_data_dict)

# Print combined data for testing purposes
loaded_data = np.load('data/combinedData.npz')
for session in loaded_data.files:
    print(loaded_data[session])