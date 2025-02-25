import numpy as np
import json
import os

def createPoseData(videoFolder: str = 'pose', dataFolder: str = 'data') -> None:
    """Given openPose JSON output, creates poseData.npz file with pose tracking data

    Args:
        videoFolder (str, optional): folder where weightlifting videos are stored. Defaults to 'pose'.
        dataFolder (str, optional): folder where .npz data files are stored. Defaults to 'data'.
    """
    # Dictionary that stores all np arrays for each session
    poseDataDict = {}
    # List of subfolders from each session with individual frame JSON keypoints data
    subfolders = os.listdir(f'{videoFolder}/')
    # Iterate through all subfolders
    for subfolder in subfolders:
        # Get all the JSON frame files from each session
        files = os.listdir(f'{videoFolder}/{subfolder}/')
        sessionNum = subfolder[7:]
        # Create empty numpy array to append data to
        poseData = np.empty(shape=[0, 77])
        # Iterate through all files in respective subfolder
        for file in files:
            file_path = f'{videoFolder}/{subfolder}/{file}'
            # Open JSON file to extract data
            with open(file_path, 'r') as json_file:
                json_data = json.load(json_file)
                frameNum = file[(file.index('_') + 1):file.index('_k')]
                # Append frame to poseData
                poseData = np.vstack((poseData, [[sessionNum, int(frameNum)] + json_data['people'][0]['pose_keypoints_2d']])).astype(np.float32)
        # Once all of the frames have been added to poseData add the numpy array for the session to the dict
        poseDataDict[sessionNum] = poseData
    # Save all arrays for each recording session into a zipped numpy file
    np.savez(f'{dataFolder}/poseData.npz', **poseDataDict)

def createSemgData(semgFolder: str = 'semg', dataFolder: str = 'data') -> None:
    """Given txt file from arduino serial output, creates semgData.npz file with semg data

    Args:
        semgFolder (str, optional): folder where txt files with sEMG data are stored. Defaults to 'semg'.
        dataFolder (str, optional): folder where .npz data files are stored. Defaults to 'data'.
    """
    # Dictionary that stores np arrays for each session
    semgDataDict = {}
    # Get all the files within the semgFolder
    files = os.listdir(f'{semgFolder}/')
    # Iterate through every file (each file is one session)
    for file in files:
        # Create empty array for session
        semgSessionData = np.empty(shape=[0,8])
        sessionNum = file[7:file.index('.')]
        # Open file with reading permissions
        with open(f'{semgFolder}/{file}', 'r') as semgFile:
            # Get list of every line
            lines = semgFile.readlines()[1:]
            # Iterate through lines and get data
            for i in range(0, len(lines), 7):
                # Create an empty array for each frame
                semgFrameData = np.empty(8)
                semgFrameData[0] = sessionNum
                # Set each value in frame array to corresponding value
                for j in range(6):
                    semgFrameData[j+1] = lines[i+j]
                semgFrameData[7] = int(lines[i+6]) - 1
                # Add frame array to 2d session array
                semgSessionData = np.vstack((semgSessionData, semgFrameData)).astype(np.int32)
        # Save session array in dict
        semgDataDict[sessionNum] = semgSessionData
    # Save all the arrays in a zipped numpy file
    np.savez(f'{dataFolder}/semgData.npz', **semgDataDict)

# TODO
def createVectorData(dataFolder: str = 'data') -> None:
    """Given poseData.npz create vectorData.npz, a file with all unit vectors between points

    Args:
        dataFolder (str, optional): folder where .npz data files are stored. Defaults to 'data'.
    """
    # Open poseData to obtain points to calculate vectors
    poseDataDict = np.load(f'{dataFolder}/poseData.npz')
    # Dictionary to store all session data
    vectorDataDict = {}
    # Beginning and end index of points being used
    minPoint, maxPoint = 1, 14
    # Iterate through every session
    for session in poseDataDict.files:
        # Create empty np array for each session (105 Vectors with x, y, confidence)
        vectorSessionData = np.empty(shape=[0,317])

# TODO Remove any mention of session number and frame number from the data
def createCombinedData(dataFolder: str = 'data') -> None:
    """Given vectorData.npz and semgData.npz file, creates combinedData.npz file with synched data

    Args:
        dataFolder (str, optional): folder where .npz data files are stored. Defaults to 'data'.
    """
    # Open vectorData and semgData
    vectorDataDict = np.load(f'{dataFolder}/vectorData.npz')
    semgDataDict = np.load(f'{dataFolder}/semgData.npz')
    

# -------------------------------- TESTING -------------------------------- #
if __name__ == '__main__':
    
    np.set_printoptions(threshold=np.inf)

    # Load poseData.npz and test
    createPoseData()
    loaded_pose_data = np.load('data/poseData.npz')
    for file in loaded_pose_data.files:
        print(loaded_pose_data[file])

    # Load semgData.npz and test
    # createSemgData()
    # loaded_semg_data = np.load('data/semgData.npz')
    # for file in loaded_semg_data.files:
    #     print(loaded_semg_data[file])

    # # Load vectorData.npz and test
    # createVectorData()
    # loaded_vector_data = np.load('data/vectorData.npz')
    # for file in loaded_vector_data.files:
    #     print(loaded_vector_data[file])

    # # Load combinedData.npz and test
    # createCombinedData()
    # loaded_combined_data = np.load('data/combinedData.npz')
    # for file in loaded_combined_data.files:
    #     print(loaded_combined_data[file])