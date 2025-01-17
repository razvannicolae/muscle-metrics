import numpy as np
import json
import os

def createPoseData():
    '''Given openPose JSON output, creates poseData.npz file with pose tracking data'''
    # Main directory for openPose JSON Outputs
    video_folder = 'pose'
    # Dictionary that stores all np arrays for each session
    pose_data_dict = {}
    # List of subfolders from each session with individual frame JSON keypoints data
    subfolders = os.listdir(f'{video_folder}/')
    # Iterate through all subfolders
    for subfolder in subfolders:
        # Get all the JSON frame files from each session
        files = os.listdir(f'{video_folder}/{subfolder}/')
        sessionNum = subfolder[7:]
        # Create empty numpy array to append data to
        poseData = np.empty(shape=[0, 77])
        # Iterate through all files in respective subfolder
        for file in files:
            file_path = f'{video_folder}/{subfolder}/{file}'
            # Open JSON file to extract data
            with open(file_path, 'r') as json_file:
                json_data = json.load(json_file)
                frameNum = file[0:file.index('_')]
                # Append frame to poseData
                poseData = np.append(poseData, [[sessionNum, str(int(frameNum))] + json_data['people'][0]['pose_keypoints_2d']], axis = 0).astype(np.float64)
        # Once all of the frames have been added to poseData add the numpy array for the session to the dict
        pose_data_dict[sessionNum] = poseData
    # Save all arrays for each recording session into a zipped numpy file
    np.savez('data/poseData.npz', **pose_data_dict)

def createSemgData():
    '''Given txt file from arduino serial output, creates semgData.npz file with semg data'''
    # Main directory for sEMG data txt files
    semg_folder = 'semg'
    # Dictionary that stores np arrays for each session
    semg_data_dict = {}
    # Get all the files within the semg_folder
    files = os.listdir(f'{semg_folder}/')
    # Iterate through every file (each file is one session)
    for file in files:
        # Create empty array for session
        semgSessionData = np.empty(shape=[0,7])
        sessionNum = file[7:file.index('.')]
        # Open file with reading permissions
        with open(f'{semg_folder}/{file}', 'r') as semgFile:
            # Get list of every line
            lines = semgFile.readlines()
            # Iterate through lines and get data
            for i in range(0, len(lines), 6):
                # Create an empty array for each frame
                semgFrameData = np.empty(7)
                semgFrameData[0] = sessionNum
                # Set each value in frame array to corresponding value
                for j in range(6):
                    semgFrameData[j+1] = lines[i+j]
                # Add frame array to 2d session array
                semgSessionData = np.vstack((semgSessionData, semgFrameData))   
        # Save session array in dict
        semg_data_dict[sessionNum] = semgSessionData
    # Save all the arrays in a zipped numpy files
    np.savez('data/semgData.npz', **semg_data_dict)

def createVectorData():
    '''Given poseData.npz create vectorData.npz, a file with all unit vectors between points'''

def createCombinedData():
    '''Given vectorData.npz and semgData.npz file, creates combinedData.npz file with synched data'''

# -------------------------------- TESTING -------------------------------- #

createPoseData()
loaded_pose_data = np.load('data/poseData.npz')
for file in loaded_pose_data.files:
    print(loaded_pose_data[file])

createSemgData()
loaded_semg_data = np.load('data/semgData.npz')
for file in loaded_semg_data.files:
    print(loaded_semg_data[file])
