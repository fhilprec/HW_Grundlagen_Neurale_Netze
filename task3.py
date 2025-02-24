import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchmetrics
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Task 3.1a: Downloading and preparing the data
print("Preparing MNIST dataset...")

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),                      # Convert images to tensors (0-1 range)
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std dev
])

# Download and prepare the test dataset
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Download and prepare the training dataset
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

print(f"Training set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")
print(f"Number of batches in training loader: {len(train_loader)}")
print(f"Number of batches in test loader: {len(test_loader)}")

# Task 3.2a: Defining the Network
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Flatten layer to convert 2D images to 1D
        self.flatten = nn.Flatten()
        
        # Input layer: 28x28 = 784 input features (image pixels)
        # First hidden layer: 25 neurons
        self.layer1 = nn.Linear(28 * 28, 25)
        self.relu1 = nn.ReLU()
        
        # Second hidden layer: 25 neurons
        self.layer2 = nn.Linear(25, 25)
        self.relu2 = nn.ReLU()
        
        # Output layer: 10 neurons (for digits 0-9)
        self.output = nn.Linear(25, 10)
    
    def forward(self, x):
        x = self.flatten(x)    # Flatten the 2D image to 1D
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x

# Task 3.2b: Using the GPU
# Select device (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model and move it to the selected device
model = Network().to(device)
print(f"Model architecture:\n{model}")

# Count model parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# Task 3.2c: Training loop
# Define loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Set number of epochs
num_epochs = 10

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
        # Move data to the selected device
        inputs, targets = inputs.to(device), targets.to(device)
        
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
plt.savefig('mnist_training_loss.png')
plt.show()

# Task 3.3a: Testing the network and calculating accuracy
print("\nTesting the network...")
model.eval()  # Set model to evaluation mode

# Initialize accuracy metric (moved to the selected device)
accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=10).to(device)

with torch.inference_mode():
    for inputs, targets in test_loader:
        # Move data to the selected device
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass to get predictions
        outputs = model(inputs)
        
        # Get the predicted class indices
        _, predicted = torch.max(outputs, 1)
        
        # Update accuracy metric
        accuracy.update(predicted, targets)

# Compute final accuracy
final_accuracy = accuracy.compute()
print(f"Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

# Task 3.3b: Saving the model parameters
model_save_path = "mnist_model.pt"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Example of loading the model
# Uncomment these lines to load the saved model
# new_model = Network().to(device)
# new_model.load_state_dict(torch.load(model_save_path))
# new_model.eval()

# Visualize some test examples with predictions
def visualize_predictions(model, device, test_loader, num_examples=10):
    # Get a batch of test data
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Move to device and make predictions
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    # Move back to CPU for visualization
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    predicted = predicted.cpu().numpy()
    
    # Plot the examples
    plt.figure(figsize=(15, 6))
    for i in range(num_examples):
        plt.subplot(2, 5, i+1)
        # Reshape and denormalize the image
        img = images[i][0]
        img = img * 0.3081 + 0.1307  # Reverse the normalization
        plt.imshow(img, cmap='gray')
        title = f"True: {labels[i]}, Pred: {predicted[i]}"
        plt.title(title, color="green" if predicted[i] == labels[i] else "red")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_predictions.png')
    plt.show()

# Visualize some test examples
print("\nVisualizing some test examples with predictions...")
visualize_predictions(model, device, test_loader)

# Calculate and display confusion matrix
def display_confusion_matrix(model, device, test_loader):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Get all predictions
    all_preds = []
    all_labels = []
    
    with torch.inference_mode():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('mnist_confusion_matrix.png')
    plt.show()
    
    # Calculate per-class accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_accuracy):
        print(f"Accuracy for digit {i}: {acc:.4f}")

# Display confusion matrix
print("\nCalculating confusion matrix...")
try:
    import seaborn
    display_confusion_matrix(model, device, test_loader)
except ImportError:
    print("Seaborn not installed. Skipping confusion matrix visualization.")