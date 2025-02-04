import subprocess
import os

def runOpenpose():
    # Directory containing videos
    videoFolder = "videos"

    # List all video files in the folder
    videoList = os.listdir(videoFolder)

    for video in videoList:
        # Construct full path for each video
        videoPath = os.path.join(os.path.abspath(videoFolder), video)

        # Ensure paths are properly quoted
        command = ["bin\OpenPoseDemo.exe", "--video", f"{videoPath}", "--write_json", os.getcwd() + f"\pose\session{videoList.index(video)}"]
        print(f"Running command: {' '.join(command)}")

        try:
            # Execute the command
            result = subprocess.run(command, cwd = "C:\\Users\\Shreyas\\Downloads\\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended\\openpose", check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            print(result.stdout.decode('utf-8'))  # Print the standard output
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while processing {video}: {e.stderr.decode('utf-8')}\n")

runOpenpose()