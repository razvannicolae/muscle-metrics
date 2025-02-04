# Define the URL and the destination file path
$url = "https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/download/v1.7.0/openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended.zip"
$zipFile = "openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended.zip"
$destinationFolder = "openpose"

# Download the zip file
Invoke-WebRequest -Uri $url -OutFile $zipFile

# Extract the zip file to the current directory
Expand-Archive -Path $zipFile -DestinationPath .

# Clean up by removing the zip file
Remove-Item -Path $zipFile

# Move every file over into the openpose folder
Get-ChildItem -Path . -Exclude $destinationFolder | ForEach-Object {
    Move-Item -Path $_.FullName -Destination $destinationFolder -Recurse -Force
}

# Create new folders within the openpose folder
$folders = @("pose", "videos", "semg", "data") | ForEach-Object { 
    Join-Path -Path $destinationFolder -ChildPath $_ 
}
New-Item -ItemType Directory -Path $folders -Force