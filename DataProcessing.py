import numpy as np
import json
import os

def create_pose_data(video_folder: str = 'pose', data_folder: str = 'data') -> None:
    """Given openPose JSON output, creates pose_data.npz file with pose tracking data
    Args:
        video_folder (str, optional): folder where weightlifting videos are stored. Defaults to 'pose'.
        data_folder (str, optional): folder where created pose_data.npz data file is saved. Defaults to 'data'.
    """
    # Dictionary that stores all np arrays for each session
    pose_data_dict = {}
    # List of subfolders from each session with individual frame JSON keypoints data
    subfolders = os.listdir(f'{video_folder}/')
    # Iterate through all subfolders
    for subfolder in subfolders:
        # Get all the JSON frame files from each session
        files = os.listdir(f'{video_folder}/{subfolder}/')
        session_num = subfolder[7:]
        # Create empty numpy array to append data to
        pose_data = np.empty(shape=[0, 77])
        # Iterate through all files in respective subfolder
        for file in files:
            file_path = f'{video_folder}/{subfolder}/{file}'
            # Open JSON file to extract data
            with open(file_path, 'r') as json_file:
                json_data = json.load(json_file)
                frame_num = file[(file.index('_') + 1):file.index('_k')]
                # Append frame to pose_data 
                pose_data = np.vstack((pose_data, [[session_num, int(frame_num)] + json_data['people'][0]['pose_keypoints_2d']])).astype(np.float32)
        # Once all of the frames have been added to pose_data add the numpy array for the session to the dict
        pose_data_dict[session_num] = pose_data
    # Save all arrays for each recording session into a zipped numpy file
    np.savez(f'{data_folder}/pose_data.npz', **pose_data_dict)

def create_semg_data(semg_folder: str = 'semg', data_folder: str = 'data') -> None:
    """Given txt file from arduino serial output, creates semgData.npz file with semg data
    Args:
        semg_folder (str, optional): folder where txt files with sEMG data are stored. Defaults to 'semg'.
        data_folder (str, optional): folder where .npz data files are stored. Defaults to 'data'.
    """
    # Dictionary that stores np arrays for each session
    semg_data_dict = {}
    # Get all the files within the semg_folder
    files = os.listdir(f'{semg_folder}/')
    # Iterate through every file (each file is one session)
    for file in files:
        # Create empty array for session
        semg_session_data = np.empty(shape=[0,8])
        session_num = file[7:file.index('.')]
        # Open file with reading permissions
        with open(f'{semg_folder}/{file}', 'r') as semg_file:
            # Get list of every line
            lines = semg_file.readlines()[1:]
            # Iterate through lines and get data
            for i in range(0, len(lines), 7):
                # Create an empty array for each frame
                semg_frame_data = np.empty(8)
                semg_frame_data[0] = session_num
                # Set each value in frame array to corresponding value
                for j in range(6):
                    semg_frame_data[j+1] = lines[i+j]
                semg_frame_data[7] = int(lines[i+6]) - 1
                # Add frame array to 2d session array
                semg_session_data = np.vstack((semg_session_data, semg_frame_data)).astype(np.float32)
        # Save session array in dict
        semg_data_dict[session_num] = semg_session_data
    # Save all the arrays in a zipped numpy file
    np.savez(f'{data_folder}/semg_data.npz', **semg_data_dict)

# TODO
def create_vector_data(data_folder: str = 'data') -> None:
    """Given pose_data.npz create vector_data.npz, a file with all unit vectors between points
    Args:
        data_folder (str, optional): folder where .npz data files are stored. Defaults to 'data'.
    """
    # Open pose_data  to obtain points to calculate vectors
    pose_data_dict = np.load(f'{data_folder}/pose_data.npz')
    # Dictionary to store all session data
    vector_data_dict = {}
    # Beginning and end index of points being used
    min_point, max_point = 1, 14
    # Iterate through every session
    for session in pose_data_dict.files:
        # Create empty np array for each session (105 Vectors with x, y, confidence)
        vector_session_data = np.empty(shape=[0,317])
        for frame in pose_data_dict[session]:
            # Create empty np array for each frame (session, frame, 105 vectors)
            vector_frame_data = np.empty(317)
            vector_frame_data[:2] = frame[:2]
            vector_index = 0
            # Iterate through every point
            for first_point in range(min_point, max_point - 1):
                for second_point in range(first_point + 1, max_point):
                    # Calculate the vector between the two points
                    x = frame[2+first_point*3] - frame[2+second_point*3]
                    y = frame[2+first_point*3 + 1] - frame[2+second_point*3 + 1]
                    # Normalize the vector
                    magnitude = np.sqrt(x**2 + y**2)
                    x /= magnitude if magnitude < 1 and x != 0 else 0
                    y /= magnitude if magnitude < 1 and y != 0 else 0
                    confidence = (frame[2+first_point*3 + 2] + frame[2+second_point*3 + 2]) / 2
                    # Add the vector to the frame array
                    vector_frame_data[2+vector_index: 5 + vector_index] = [x, y, confidence]
                    vector_index += 3
            # Add the frame array to the session array
            vector_session_data = np.vstack((vector_session_data, vector_frame_data)).astype(np.float32)
        # Save the session data in the dictionary
        vector_data_dict[session] = vector_session_data
    # Save all the arrays in a zipped numpy file
    np.savez(f'{data_folder}/vector_data.npz', **vector_data_dict)
    
if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)

    # # Load pose_data.npz and test
    # create_pose_data()
    # loaded_pose_data = np.load('data/pose_data.npz')
    # for file in loaded_pose_data.files:
    #     print(loaded_pose_data[file])

    # # Load semg_data.npz and test
    # create_semg_data()
    # loaded_semg_data = np.load('data/semg_data.npz')
    # for file in loaded_semg_data.files:
    #     print(loaded_semg_data[file])

    # Load vectorData.npz and test
    # create_vector_data()
    loaded_vector_data = np.load('data/vector_data.npz')
    for file in loaded_vector_data.files:
        print(loaded_vector_data[file])