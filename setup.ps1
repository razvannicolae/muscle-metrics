# Define the URL for the OpenPose zip file
$openPoseUrl = "https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/download/v1.7.0/openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended.zip"

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

# Move all files from the extracted directory to the repository root, avoiding errors
Get-ChildItem -Path $extractionPath -Recurse | ForEach-Object {
    try {
        Move-Item -Path $_.FullName -Destination $PSScriptRoot -Force -ErrorAction Stop
    } catch {
        Write-Host "Error moving item: $($_.FullName)"
    }
}

# Remove the downloaded zip file and the extraction directory
Remove-Item -Path $zipFilePath -Force
Remove-Item -Path $extractionPath -Recurse -Force

# Create the required folders
$folders = @("folder1", "folder2", "folder3", "folder4")
foreach ($folder in $folders) {
    $folderPath = "$PSScriptRoot\$folder"
    if (-Not (Test-Path -Path $folderPath)) {
        New-Item -ItemType Directory -Path $folderPath
    }
}

# Create the classification.json file
$classJsonPath = "$PSScriptRoot\classification.json"
if (-Not (Test-Path -Path $classJsonPath)) {
    New-Item -ItemType File -Path $classJsonPath
}

# Create the .env file
$envFilePath = "$PSScriptRoot\.env"
if (-Not (Test-Path -Path $envFilePath)) {
    New-Item -ItemType File -Path $envFilePath
}