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
                try:
                    for j in range(6):
                        semg_frame_data[j+1] = lines[i+j]
                    if int(session_num) <= 14:
                        semg_frame_data[7] = int(lines[i+6]) - 1
                    else:
                        semg_frame_data[7] = int(lines[i+6]) 
                except IndexError:
                    break             
                # Add frame array to 2d session array
                semg_session_data = np.vstack((semg_session_data, semg_frame_data)).astype(np.float32)
        # Save session array in dict
        semg_data_dict[session_num] = semg_session_data
    # Save all the arrays in a zipped numpy file
    np.savez(f'{data_folder}/semg_data.npz', **semg_data_dict)

def create_classification_data(file_name: str = "classification.json" ) -> None:
    """Creates a numpy zipped file with form classification data for second model
    Args:
        file_name (str, optional): file path for json dictionary file. Defaults to "classification.json":str.
    """
    # Open JSON file to extract data
    with open(file_name, 'r') as json_file:
        classification_data = json.load(json_file)
    # Create empty numpy array to append data to
    classification_np = np.empty(shape=[0, 3])
    # Iterate through all the data in the JSON file
    for session in classification_data:
        classification_np = np.vstack((classification_np, [session, 1 if classification_data[session] == 1 else 0, 1 if classification_data[session] == 0 else 0])).astype(np.int8)
    # Save the numpy array into a zipped numpy file
    np.savez('data/classification_data.npz', classification_np)

def create_vector_data(data_folder: str = 'data') -> None:
    """Given pose_data.npz create vector_data.npz, a file with all unit vectors between points
    Args:
        data_folder (str, optional): folder where .npz data files are stored. Defaults to 'data'.
    """
    # Open pose_data  to obtain points to calculate vectors
    pose_data_dict = np.load(f'{data_folder}/pose_data.npz')
    # Dictionary to store all session data
    vector_data_dict = {}
    # points = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13] (point # on openPose)
    point_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12] # adjusted for array indexing
    # Iterate through every session
    for session in pose_data_dict.files:
        # Create empty np array for each session (66 Vectors with x, y, confidence)
        vector_session_data = np.empty(shape=[0,200])
        for frame in pose_data_dict[session]:
            # Create empty np array for each frame (session, frame, 66 vectors)
            vector_frame_data = np.empty(200)
            vector_frame_data[:2] = frame[:2]
            vector_index = 0
            # Iterate through every point
            for idx, first_point in enumerate(point_indexes[:-1]):
                for second_point in point_indexes[idx+1:]:
                    x1, x2 = frame[2+first_point*3], frame[2+second_point*3] # Get x values
                    y1, y2 = frame[2+first_point*3 + 1], frame[2+second_point*3 + 1] # Get y values
                    c1, c2 = frame[2+first_point*3 + 2], frame[2+second_point*3 + 2] # Get confidence values
                    # Check if points are valid
                    if not 0 in [x1, x2, y1, y2, c1, c2]:
                        x = x1 - x2
                        y = y1 - y2
                        # Normalize the vector
                        magnitude = np.sqrt(x**2 + y**2)
                        if magnitude > 1:
                            x /= magnitude
                            y /= magnitude
                            confidence = (c1 + c2) / 2
                        else: x, y, confidence = 0, 0, 0
                        # Add the vector to the frame array
                        vector_frame_data[2+vector_index: 5 + vector_index] = [x, y, confidence]
                    else: vector_frame_data[2+vector_index: 5 + vector_index] = [0, 0, 0]
                    vector_index += 3
            # Add the frame array to the session array
            vector_session_data = np.vstack((vector_session_data, vector_frame_data))
            # Replace small values with 0
            vector_session_data = np.nan_to_num(vector_session_data, nan = 0, posinf = None, neginf = None)
        # Save the session data in the dictionary
        vector_data_dict[session] = vector_session_data
    # Save all the arrays in a zipped numpy file
    np.savez(f'{data_folder}/vector_data.npz', **vector_data_dict)

if __name__ == '__main__':
    create_pose_data()
    create_semg_data()
    create_classification_data()
    create_vector_data()