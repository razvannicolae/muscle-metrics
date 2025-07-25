import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# ---------------------------
# Dataset for EMG to Form Classification
# ---------------------------
class EMGFormDataset(Dataset):
    def __init__(self, semg_file, classification_file):
        self.semg_npz = np.load(semg_file)
        self.classification_npz = np.load(classification_file)
        self.classification_data = self.classification_npz['arr_0']
        
        # Filter sessions between 15-34 and 44-65
        self.session_keys = [key for key in self.semg_npz.keys() if 15 <= int(key) <= 34 or 44 <= int(key) <= 65]

    def __len__(self):
        return len(self.session_keys)
   
    def __getitem__(self, idx):
        session_key = self.session_keys[idx]
        semg_session = self.semg_npz[session_key]  # expected shape: (300, total_columns)
        semg_session = semg_session[0:300, 1:2]   # shape: (300, 1)
        
        # Extract corresponding classification label
        session_num = int(session_key)
        classification_label = self.classification_data[session_num, 1:3]  # shape: (2,)

        # Convert to torch tensors.
        semg_tensor = torch.tensor(semg_session, dtype=torch.float32)
        class_tensor = torch.tensor(classification_label, dtype=torch.float32)
        
        return semg_tensor, class_tensor
# ---------------------------
# Model Architecture
# ---------------------------
class EMGToFormClassifier(nn.Module):
    def __init__(self):
        super(EMGToFormClassifier, self).__init__()
        self.fc1 = nn.Linear(300, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, 2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.sigmoid(self.fc_out(x))  # Sigmoid for binary classification
        return x

# ---------------------------
# Data Preparation and Split
# ---------------------------
semg_file = 'data/semg_data.npz'
classification_file = 'data/classification_data.npz'
dataset = EMGFormDataset(semg_file, classification_file)

# Split dataset into 70% training, 15% validation, 15% testing
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
# Training Setup with Class Balancing
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EMGToFormClassifier().to(device)

# Compute class weights for balanced loss
labels = dataset.classification_data[:, 1:3]
class_counts = labels.sum(axis=0)
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)

# ---------------------------
# Add Class Weight Distribution Chart
# ---------------------------
def plot_class_weight_distribution(class_counts, class_weights):
    class_names = ['Proper Form', 'Poor Form']
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # First subplot: Class counts
    ax1.bar(class_names, class_counts, color=['#3498db', '#e74c3c'])
    ax1.set_title('Class Distribution in Dataset')
    ax1.set_ylabel('Number of Samples')
    ax1.set_ylim(0, max(class_counts) * 1.2)
    
    # Add count values on top of bars
    for i, count in enumerate(class_counts):
        ax1.text(i, count + (max(class_counts) * 0.05), f"{int(count)}", 
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Second subplot: Class weights
    weights_np = class_weights.cpu().numpy()
    ax2.bar(class_names, weights_np, color=['#3498db', '#e74c3c'])
    ax2.set_title('Class Weights for Balanced Loss')
    ax2.set_ylabel('Weight Value')
    ax2.set_ylim(0, max(weights_np) * 1.2)
    
    # Add weight values on top of bars
    for i, weight in enumerate(weights_np):
        ax2.text(i, weight + (max(weights_np) * 0.05), f"{weight:.6f}", 
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add a title for the entire figure
    fig.suptitle('Class Weight Distribution Analysis', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
    
    # Save the figure
    plt.savefig('class_weight_distribution.png')
    print("Class weight distribution chart saved as 'class_weight_distribution.png'")
    plt.show()

# # Plot the class weight distribution
# plot_class_weight_distribution(class_counts, class_weights)

criterion = nn.BCELoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Function to apply winner-takes-all to predictions
def winner_takes_all(predictions) -> torch.Tensor:

    # Get indices of maximum values along dimension 1
    max_indices = torch.argmax(predictions, dim=1, keepdim=True)
    
    # Create a tensor of zeros with the same shape as predictions
    one_hot = torch.zeros_like(predictions)
    
    # Set the maximum value positions to 1
    one_hot.scatter_(1, max_indices, 1.0)
    
    return one_hot

for epoch in range(num_epochs):
    # Training Phase
    model.train()
    running_train_loss = 0.0
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
        predictions = winner_takes_all(outputs)
        correct_train += (predictions == class_tensor).all(dim=1).sum().item()
        total_train += class_tensor.size(0)
    
    train_acc = correct_train / total_train
    train_losses.append(running_train_loss / len(train_loader))
    train_accuracies.append(train_acc)
    
    # Validation Phase
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for semg_tensor, class_tensor in val_loader:
            semg_tensor, class_tensor = semg_tensor.to(device), class_tensor.to(device)
            outputs = model(semg_tensor)
            loss = criterion(outputs, class_tensor)
            running_val_loss += loss.item()
            
            predictions = winner_takes_all(outputs)
            correct_val += (predictions == class_tensor).all(dim=1).sum().item()
            total_val += class_tensor.size(0)
    
    val_acc = correct_val / total_val
    val_losses.append(running_val_loss / len(val_loader))
    val_accuracies.append(val_acc)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_losses[-1]:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_losses[-1]:.4f} - Val Acc: {val_acc:.4f}")

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

        predictions = winner_takes_all(outputs)
        all_predictions.append(predictions.cpu().numpy())
        all_labels.append(class_tensor.cpu().numpy())
        
        correct_test += (predictions == class_tensor).all(dim=1).sum().item()
        total_test += class_tensor.size(0)
        print(f"Epoch {epoch+1}: Sample Predictions: {predictions} | True Labels: {class_tensor}")


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
torch.save(model.state_dict(), 'saved_models/emg_form_classifier.pth')
print("Model saved as emg_form_classifier.pth")