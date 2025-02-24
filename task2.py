import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Task 2.1: Our Dataset
print("Downloading penguin dataset...")
penguins = fetch_openml(name='penguins', parser="auto", as_frame=True).frame
print(f"Dataset shape: {penguins.shape}")

# Remove rows with missing values
penguins_clean = penguins.dropna(axis=0)
print(f"Shape after removing NA values: {penguins_clean.shape}")

# Select features and target
target = penguins_clean['species']
features = penguins_clean[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']]

# Task 2.1a: Data preparation for neural Networks
# 1. Encode the target as numbers using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(target)
print(f"Original classes: {label_encoder.classes_}")
print(f"Encoded classes: {np.unique(y_encoded)}")

# 2. Scale features to [0,1] range using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(features)
print(f"Feature ranges before scaling: {features.min().values} to {features.max().values}")
print(f"Feature ranges after scaling: {X_scaled.min(axis=0)} to {X_scaled.max(axis=0)}")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Task 2.1b: PyTorch DataLoader
class SklearnDataSet(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()  # Use long for classification targets
        
    def __len__(self):
        return self.x.size(dim=0)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Create dataset objects
data_train = SklearnDataSet(X_train, y_train)
data_test = SklearnDataSet(X_test, y_test)

# Create DataLoaders
# For training: batch size 5 and shuffling enabled
train_loader = DataLoader(data_train, batch_size=5, shuffle=True)
# For testing: default settings
test_loader = DataLoader(data_test, batch_size=len(data_test))

print(f"Number of batches in training loader: {len(train_loader)}")
print(f"Number of batches in test loader: {len(test_loader)}")

# Task 2.2a: Defining a Network
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Input layer has 4 features (culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g)
        # First hidden layer has 4 neurons
        self.layer1 = nn.Linear(4, 4)
        self.relu1 = nn.ReLU()
        
        # Second hidden layer has 4 neurons
        self.layer2 = nn.Linear(4, 4)
        self.relu2 = nn.ReLU()
        
        # Output layer has 3 neurons (one for each penguin species)
        self.output = nn.Linear(4, 3)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x

# Initialize the model
model = Network()
print(f"Model architecture:\n{model}")

# Task 2.2b: Training loop
# Define loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Set number of epochs
num_epochs = 50  # You can adjust this as needed

# Lists to store epoch losses for plotting
epoch_numbers = []
epoch_losses = []

# Training loop
print("\nTraining the network...")
model.train()  # Set model to training mode
for epoch in range(num_epochs):
    running_loss = 0.0
    batch_count = 0
    
    for inputs, targets in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item()
        batch_count += 1
    
    # Calculate average loss for this epoch
    avg_loss = running_loss / batch_count
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Store for plotting
    epoch_numbers.append(epoch + 1)
    epoch_losses.append(avg_loss)

print("Training complete!")

# Plot the training loss across epochs
plt.figure(figsize=(10, 6))
plt.plot(epoch_numbers, epoch_losses, 'b-')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss Over Time')
plt.grid(True)
plt.savefig('training_loss.png')
plt.show()

# Task 2.3a: Testing the network and calculating accuracy
print("\nTesting the network...")
model.eval()  # Set model to evaluation mode

# Initialize accuracy metric
accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=3)

with torch.inference_mode():
    for inputs, targets in test_loader:
        # Forward pass to get predictions
        outputs = model(inputs)
        
        # Get the predicted class indices
        _, predicted = torch.max(outputs, 1)
        
        # Update accuracy metric
        accuracy.update(predicted, targets)

# Compute final accuracy
final_accuracy = accuracy.compute()
print(f"Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

# Task 2.3b: Accessing the model parameters
print("\nModel parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data.shape}")

print("\nComplete model state dictionary:")
print(model.state_dict())

# Optional: Visualize predictions
with torch.inference_mode():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        # Convert back to original class names
        predicted_species = [label_encoder.classes_[i] for i in predicted.numpy()]
        actual_species = [label_encoder.classes_[i] for i in targets.numpy()]
        
        # Display sample of predictions
        print("\nSample predictions (first 10):")
        print("Predicted\tActual")
        for p, a in zip(predicted_species[:10], actual_species[:10]):
            print(f"{p}\t{a}")
        
        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(targets.numpy(), predicted.numpy())
        print("\nConfusion Matrix:")
        print(cm)
        
        # No need to continue after the first batch
        break