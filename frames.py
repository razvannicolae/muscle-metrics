import numpy as np
import json
import os

"""
Data Structure:
openpose/
  ├── data/
  │   ├── combineddata.npz      # Combined pose keypoints and sEMG data
  │   │   ├── array1            # Data for first session
  │   │   └── ...              
  │   └── framedata.npz         # Just pose keypoint data
  │       ├── array1            # Data for first session
  │       └── ...                
  │
  ├── frames/                   # Individual frame data
  │   ├── frame1/               # Data for first session
  │   │   ├── 000000000000_keypoints.json  # Keypoints for frame 0
  │   │   ├── 000000000001_keypoints.json  # Keypoints for frame 1
  │   │   └── ...                          
  │   ├── frame2/               # Data for second session
  │   │   ├── 000000000000_keypoints.json
  │   │   ├── 000000000001_keypoints.json
  │   │   └── ...
  │   └── ...                   # Additional sessions
  │
  └── semg/                     # sEMG data files
      ├── session1.txt          # sEMG data for first session
      ├── session2.txt          # sEMG data for second session
      └── ...                   # Additional session files

Data Specifications:
- combineddata.npz contains synchronized pose keypoints and sEMG data
- framedata.npz contains only the pose keypoint sequences
- Each frame folder contains sequential JSON files with pose keypoints
- JSON files follow OpenPose output format for keypoint data
- semg folder contains individual text files for each session's sEMG readings
- .npz files have a U32 dtype, will be converted when inputted into PyTorch
"""

# Iterate through all subfolders
def createFile(video_folder):

    frame_data_dict = {}

    # List of subfolders with individual frame JSON keypoints data
    subfolders = os.listdir(video_folder)

    for subfolder in subfolders:

        files = os.listdir(f"{video_folder}/{subfolder}/")
        folderNum = f'array{subfolder[6:]}'
        frameData = np.empty(shape=[0, 77])

        # Iterate through all files in respective subfolder
        for file in files:

            file_path = f"{video_folder}/{subfolder}/{file}"

            # Open JSON file to extract data
            with open(file_path, 'r') as json_file:

                json_data = json.load(json_file)
                frameNum = file[0:file.index("_")]
                frameData = np.append(frameData, [[folderNum, frameNum] + json_data["people"][0]["pose_keypoints_2d"]], axis = 0)

        frame_data_dict[folderNum] = frameData

    # Save all arrays for each recording session into a zipped numpy file
    np.savez('data/frameData.npz', **frame_data_dict)


createFile('frames')

# Load data from npz file add sEMG readings and save again
loaded_data = np.load('data/frameData.npz')
combined_data_dict = {}
for session in loaded_data.files:
    semgData = np.random.rand(loaded_data[session].shape[0], 5)
    combined_data_dict[session] = np.hstack((loaded_data[session], semgData))
np.savez('data/combinedData.npz', **combined_data_dict)

# Print combined data
loaded_data = np.load('data/combinedData.npz')
for session in loaded_data.files:
    print(session)
    print(loaded_data[session])