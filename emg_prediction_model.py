import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# ---------------------------
# Dataset for Pose and SEMG Data (Session-wise)
# ---------------------------
class PoseSemgDataset(Dataset):
    def __init__(self, pose_file, semg_file, fps=30, duration=10):
        self.pose_npz = np.load(pose_file)
        self.semg_npz = np.load(semg_file)
       
        # Assume both files have the same session keys.
        self.session_keys = list(self.pose_npz.keys())
        self.fps = fps
        self.duration = duration  # seconds
        self.num_frames = self.fps * self.duration  # e.g. 300 frames for 10 seconds

    def __len__(self):
        return len(self.session_keys)
   
    def __getitem__(self, idx):
        session_key = self.session_keys[idx]
        pose_session = self.pose_npz[session_key]  # expected shape: (300, total_columns)
        semg_session = self.semg_npz[session_key]    # expected shape: (300, total_columns)
       
        # Extract relevant columns:
        # For pose: columns 2 to 76 → 75 features per frame.
        pose_session = pose_session[0:300, 2:77]  # shape: (300, 75)
        # For SEMG: column 0 → 1 sensor value per frame.
        semg_session = semg_session[0:300, 1:2]   # shape: (300, 1)

        if pose_session.shape[0] != self.num_frames or semg_session.shape[0] != self.num_frames:
            raise ValueError(f"Session {session_key} does not have {self.num_frames} frames.")
       
        # Reshape pose data: (300, 75) -> (fps, duration, 75) -> then transpose to (75, fps, duration)
        pose_session = pose_session.reshape(self.fps, self.duration, 75).transpose(2, 0, 1)
       
        # Reshape SEMG data: (300, 1) -> (fps, duration, 1)
        semg_session = semg_session.reshape(self.fps, self.duration, 1)
       
        # Convert to torch tensors.
        pose_tensor = torch.tensor(pose_session, dtype=torch.float32)
        semg_tensor = torch.tensor(semg_session, dtype=torch.float32)
       
        return pose_tensor, semg_tensor

# ---------------------------
# Model Architecture
# ---------------------------
class EMGEstimator(nn.Module):
    def __init__(self):
        super(EMGEstimator, self).__init__()
        # Input: (batch, 75, 30, 10)
        # Convolve over the fps dimension (30 frames per second) with kernel size (30, 1)
        self.conv = nn.Conv2d(in_channels=75, out_channels=75, kernel_size=(30, 1))
        # After conv: shape becomes (batch, 75, 1, 10) → squeeze to (batch, 75, 10)
        # Flattened feature size: 75 * 10 = 750
        self.fc_in = nn.Linear(750, 64)
        self.fc_2 = nn.Linear(64, 64)
        self.fc_3 = nn.Linear(64, 64)
        self.fc_4 = nn.Linear(64, 64)
        # Final layer: Output shape = 30 (fps) * 10 (seconds) * 1 (sensor) = 300
        self.fc_out = nn.Linear(64, 300)
   
    def forward(self, x):
        x = self.conv(x)         # -> (batch, 75, 1, 10)
        x = x.squeeze(2)         # -> (batch, 75, 10)
        x = x.view(x.size(0), -1) # Flatten to (batch, 750)
        x = F.relu(self.fc_in(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        x = F.relu(self.fc_4(x))
        x = self.fc_out(x)          # -> (batch, 300)
        x = x.view(x.size(0), 30, 10, 1)  # Reshape to (batch, 30, 10, 1)
        return x

# ---------------------------
# Data Preparation and Split
# ---------------------------
pose_file = 'data\poseData.npz'
semg_file = 'data\semgData.npz'
dataset = PoseSemgDataset(pose_file, semg_file)

# Split dataset into 70% training, 15% validation, 20% testing
dataset_length = len(dataset)
train_size = int(0.7 * dataset_length)
val_size = int(0.15 * dataset_length)
test_size = dataset_length - train_size - val_size


# Split the dataset randomly into training, validation, and testing subsets
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for each split
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ---------------------------
# Training Setup
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EMGEstimator().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training Phase
    model.train()
    running_train_loss = 0.0
    for pose_tensor, semg_tensor in train_loader:
        pose_tensor, semg_tensor = pose_tensor.to(device), semg_tensor.to(device)
        optimizer.zero_grad()           # Reset gradients
        outputs = model(pose_tensor)      # Forward pass
        loss = criterion(outputs, semg_tensor)  # Compute loss
        loss.backward()                 # Backpropagation
        optimizer.step()                # Update weights
        
        running_train_loss += loss.item()
    
    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation Phase
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for pose_tensor, semg_tensor in val_loader:
            pose_tensor, semg_tensor = pose_tensor.to(device), semg_tensor.to(device)
            outputs = model(pose_tensor)
            loss = criterion(outputs, semg_tensor)
            running_val_loss += loss.item()
    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

# ---------------------------
# Testing Phase
# ---------------------------
model.eval()
running_test_loss = 0.0
with torch.no_grad():
    for pose_tensor, semg_tensor in test_loader:
        pose_tensor, semg_tensor = pose_tensor.to(device), semg_tensor.to(device)
        outputs = model(pose_tensor)
        loss = criterion(outputs, semg_tensor)
        running_test_loss += loss.item()
avg_test_loss = running_test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")

# ---------------------------
# Plotting Loss Curves
# ---------------------------
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss Over Epochs")
plt.show()
    

torch.save(model.state_dict(), 'emg_estimator_model_10sec_1sensor.pth')
print("Model saved as emg_estimator_model_10sec_1sensor.pth")