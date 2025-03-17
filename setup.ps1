# Define the URL for the OpenPose zip file
$openPoseUrl = "https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/download/v1.7.0/openpose-1.7.0-binaries-win64.zip"

# Define the path for the downloaded zip file
$zipFilePath = "$PSScriptRoot\openpose.zip"

# Define the extraction path
$extractionPath = "$PSScriptRoot\openpose"

# Download the OpenPose zip file
Invoke-WebRequest -Uri $openPoseUrl -OutFile $zipFilePath

# Create the extraction directory if it doesn't exist
if (-Not (Test-Path -Path $extractionPath)) {
    New-Item -ItemType Directory -Path $extractionPath
}

# Unzip the downloaded file
Expand-Archive -Path $zipFilePath -DestinationPath $extractionPath

# Move all files from the extracted directory to the repository root
Get-ChildItem -Path $extractionPath -Recurse | Move-Item -Destination $PSScriptRoot -Force

# Remove the downloaded zip file and the extraction directory
Remove-Item -Path $zipFilePath -Force
Remove-Item -Path $extractionPath -Recurse -Force

Write-Output "OpenPose has been downloaded, extracted, and moved to the repository directory."