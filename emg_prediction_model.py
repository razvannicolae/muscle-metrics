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
        pose_session = pose_session[0:300, 2:200]  # shape: (300, 75)
        # For SEMG: column 0 → 1 sensor value per frame.
        semg_session = semg_session[0:300, 5:6]   # shape: (300, 1)

        if pose_session.shape[0] != self.num_frames or semg_session.shape[0] != self.num_frames:
            raise ValueError(f"Session {session_key} does not have {self.num_frames} frames.")
       
        # Reshape pose data: (300, 75) -> (fps, duration, 75) -> then transpose to (75, fps, duration)
        pose_session = pose_session.reshape(self.fps, self.duration, 198).transpose(2, 0, 1)
       
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
        self.conv = nn.Conv2d(in_channels=198, out_channels=198, kernel_size=(30, 1))
        # After conv: shape becomes (batch, 75, 1, 10) → squeeze to (batch, 75, 10)
        # Flattened feature size: 75 * 10 = 750
        self.fc_in = nn.Linear(1980, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, 128)
        self.fc_4 = nn.Linear(128, 128)
        # Final layer: Output shape = 30 (fps) * 10 (seconds) * 1 (sensor) = 300
        self.fc_out = nn.Linear(128, 300)
   
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
<<<<<<< HEAD
semg_file = 'data/semg_data.npz'
classification_file = 'data/classification_data.npz'
dataset = EMGFormDataset(semg_file, classification_file)

# Split dataset into 70% training, 15% validation, 15% testing
=======
<<<<<<<< HEAD:model.py
pose_file = 'data\\vector_data.npz'
========
pose_file = 'data\pose_data.npz'
semg_file = 'data\semg_data.npz'
dataset = PoseSemgDataset(pose_file, semg_file)

# Split dataset into 70% training, 15% validation, 20% testing
dataset_length = len(dataset)
train_size = int(0.7 * dataset_length)
val_size = int(0.15 * dataset_length)
test_size = dataset_length - train_size - val_size

<<<<<<< HEAD
=======

# Split the dataset randomly into training, validation, and testing subsets
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for each split
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ---------------------------
<<<<<<< HEAD
# Training Setup with Class Balancing
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EMGToFormClassifier().to(device)

# Compute class weights for balanced loss
labels = dataset.classification_data[:, 1:3]
class_counts = labels.sum(axis=0)
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)

criterion = nn.BCELoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
=======
# Training Setup
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EMGEstimator().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 2

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training Phase
    model.train()
    running_train_loss = 0.0
<<<<<<< HEAD
    correct_train = 0
    total_train = 0
    
    for semg_tensor, class_tensor in train_loader:
        semg_tensor, class_tensor = semg_tensor.to(device), class_tensor.to(device)
        optimizer.zero_grad()
        outputs = model(semg_tensor)
        loss = criterion(outputs, class_tensor)
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item()
        predictions = (outputs > 0.5).float()
        correct_train += (predictions == class_tensor).all(dim=1).sum().item()
        total_train += class_tensor.size(0)
    
    train_acc = correct_train / total_train
    train_losses.append(running_train_loss / len(train_loader))
    train_accuracies.append(train_acc)
=======
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
<<<<<<< HEAD
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for semg_tensor, class_tensor in val_loader:
            semg_tensor, class_tensor = semg_tensor.to(device), class_tensor.to(device)
            outputs = model(semg_tensor)
            loss = criterion(outputs, class_tensor)
            running_val_loss += loss.item()
            
            predictions = (outputs > 0.5).float()
            correct_val += (predictions == class_tensor).all(dim=1).sum().item()
            total_val += class_tensor.size(0)
    
    val_acc = correct_val / total_val
    val_losses.append(running_val_loss / len(val_loader))
    val_accuracies.append(val_acc)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_losses[-1]:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_losses[-1]:.4f} - Val Acc: {val_acc:.4f}")
=======
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
<<<<<<< HEAD
from sklearn.metrics import precision_score, recall_score, f1_score

# ---------------------------
# Testing Phase
# ---------------------------
model.eval()
running_test_loss = 0.0
correct_test = 0
total_test = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for semg_tensor, class_tensor in test_loader:
        semg_tensor, class_tensor = semg_tensor.to(device), class_tensor.to(device)
        outputs = model(semg_tensor)
        loss = criterion(outputs, class_tensor)
        running_test_loss += loss.item()

        predictions = (outputs > 0.5).float()
        all_predictions.append(predictions.cpu().numpy())
        all_labels.append(class_tensor.cpu().numpy())
        
        correct_test += (predictions == class_tensor).all(dim=1).sum().item()
        total_test += class_tensor.size(0)
        print(f"Epoch {epoch+1}: Sample Predictions: {predictions[:5]} | True Labels: {class_tensor[:5]}")


# Flatten the lists for precision, recall, and F1 calculation
all_predictions = np.concatenate(all_predictions, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Calculate Precision, Recall, and F1-Score
precision = precision_score(all_labels, all_predictions, average='macro')
recall = recall_score(all_labels, all_predictions, average='macro')
f1 = f1_score(all_labels, all_predictions, average='macro')

test_acc = correct_test / total_test
print(f"Test Loss: {running_test_loss / len(test_loader):.4f} - Test Acc: {test_acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")


# ---------------------------
# Save Model
# ---------------------------
torch.save(model.state_dict(), 'emg_form_classifier.pth')
print("Model saved as emg_form_classifier.pth")
=======
# Testing Phase
model.eval()
running_test_loss = 0.0
with torch.no_grad():
    for pose_tensor, semg_tensor in test_loader:
        pose_tensor, semg_tensor = pose_tensor.to(device), semg_tensor.to(device)
        outputs = model(pose_tensor)
        loss = criterion(outputs, semg_tensor)
        running_test_loss += loss.item()
avg_test_loss = running_test_loss / len(test_loader)
avg_test_rmse = np.sqrt(avg_test_loss)
print(f"Test Loss: {avg_test_loss:.4f} - Test RMSE: {avg_test_rmse:.4f}")

# ---------------------------
# Plotting Loss Curves
# ---------------------------
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses, label="Train Loss")
# plt.plot(val_losses, label="Validation Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.title("Training and Validation Loss Over Epochs")
# plt.show()
    
sample_pose, sample_semg = next(iter(test_loader))
sample_pose = sample_pose.to(device)
model.eval()
with torch.no_grad():
    sample_pred = model(sample_pose)
   
# sample_pred and sample_semg shape: (1, 30, 10, 1)
# Flatten the time dimensions to get a (300,) array.
sample_pred = sample_pred.cpu().view(-1, 1).numpy()   # shape: (300, 1)
sample_semg = sample_semg.cpu().view(-1, 1).numpy()     # shape: (300, 1)

# Create a time axis (0 to 299)
time_steps = range(sample_pred.shape[0])

# Plot ground truth vs predicted for the one sensor.
plt.figure(figsize=(12, 4))
plt.plot(time_steps, sample_semg[:, 0], label="Ground Truth", color="blue")
plt.plot(time_steps, sample_pred[:, 0], label="Predicted", color="red", linestyle="--")
plt.title("SEMG Sensor: Ground Truth vs Prediction")
plt.xlabel("Frame")
plt.ylabel("Sensor Value")
plt.legend()
plt.tight_layout()
plt.show()



torch.save(model.state_dict(), 'saved_models/emg_estimator_model_10sec_1sensor.pth')
print("Model saved as emg_estimator_model_10sec_1sensor.pth")
