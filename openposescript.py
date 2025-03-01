import subprocess
import os

def run_openpose(video_folder: str = "videos", pose_folder: str = "pose") -> None:
    """Run Openpose on all videos in the video folder and save the pose data in the pose folder
    Args:
        videoFolder (str, optional): folder where weightlifting videos are stored. Defaults to "videos".
        poseFolder (str, optional): folder where JSON pose output files are saved to. Defaults to "pose".
    """
    # List all video files in the folder
    video_list = os.listdir(video_folder)
    for video in video_list:
        # Construct full path for each video
        video_path = os.path.join(os.path.abspath(video_folder), video)
        # Ensure paths are properly quoted
        command = ["bin\OpenPoseDemo.exe", "--video", f"{video_path}", "--write_json", os.getcwd() + f"\{pose_folder}\session{video_list.index(video)}", "--number_people_max", "1"]
        print(f"Running command: {' '.join(command)}")
        try:
            cwd = os.getenv("OPENPOSE_PATH")
            # Execute the command
            result = subprocess.run(command, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            print(result.stdout.decode('utf-8'))  # Print the standard output
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while processing {video}: {e.stderr.decode('utf-8')}\n")

if __name__ == "__main__":
    run_openpose()
