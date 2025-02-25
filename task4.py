import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchmetrics
import matplotlib.pyplot as plt
import numpy as np
import copy

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define network architecture
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Flatten layer to convert 2D images to 1D
        self.flatten = nn.Flatten()
        
        # Input layer: 28x28 = 784 input features (image pixels)
        # First hidden layer: 25 neurons
        self.fc1 = nn.Linear(28 * 28, 25)
        self.relu1 = nn.ReLU()
        
        # Second hidden layer: 25 neurons
        self.fc2 = nn.Linear(25, 25)
        self.relu2 = nn.ReLU()
        
        # Output layer: 10 neurons (for digits 0-9)
        self.fc3 = nn.Linear(25, 10)
    
    def forward(self, x):
        x = self.flatten(x)    # Flatten the 2D image to 1D
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Function to load MNIST dataset and create dataloaders
def load_mnist_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Create dataloaders
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
    
    return train_loader, test_loader

# Function to evaluate model accuracy
def evaluate_model(model, test_loader, device):
    model.eval()
    accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=10).to(device)
    
    with torch.inference_mode():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            accuracy.update(predicted, targets)
    
    return accuracy.compute()

# Function to train or finetune model
def train_model(model, train_loader, device, epochs, optimizer, loss_function):
    model.train()
    epoch_losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        batch_count = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
        
        avg_loss = running_loss / batch_count
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return epoch_losses

# Main function for Task 4.1: Pruning after training
def prune_after_training():
    print("\n=== Task 4.1: Pruning after training ===")
    
    # Load data
    train_loader, test_loader = load_mnist_data()
    
    # Select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Task 4.1a: Load network
    print("\nLoading model...")
    model = Network().to(device)
    
    # Try to load saved model, or train one if not available
    try:
        model.load_state_dict(torch.load('mnist_model.pt'))
        print("Loaded saved model from mnist_model.pt")
    except:
        print("No saved model found, training new model...")
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        train_model(model, train_loader, device, epochs=10, optimizer=optimizer, loss_function=loss_function)
        torch.save(model.state_dict(), 'mnist_model.pt')
    
    # Evaluate baseline accuracy
    baseline_accuracy = evaluate_model(model, test_loader, device)
    print(f"Baseline accuracy: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    
    # Task 4.1b: Prune the model
    print("\nPruning 60% of weights...")
    parameters_to_prune = (
        (model.fc1, 'weight'),
        (model.fc2, 'weight'),
        (model.fc3, 'weight'),
    )
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.6,
    )
    
    # Calculate the sparsity after pruning
    def calculate_sparsity(model):
        total_params = 0
        pruned_params = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                mask = module.weight_mask
                total_params += mask.nelement()
                pruned_params += torch.sum(mask == 0).item()
        return pruned_params / total_params if total_params > 0 else 0
    
    pruned_sparsity = calculate_sparsity(model)
    print(f"Sparsity after pruning: {pruned_sparsity:.4f} ({pruned_sparsity*100:.2f}%)")
    
    # Evaluate pruned model
    pruned_accuracy = evaluate_model(model, test_loader, device)
    print(f"Accuracy after pruning: {pruned_accuracy:.4f} ({pruned_accuracy*100:.2f}%)")
    print(f"Accuracy decrease: {(baseline_accuracy - pruned_accuracy)*100:.2f}%")
    
    # Task 4.1c: Finetune pruned model
    print("\nFinetuning pruned model...")
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Lower learning rate for finetuning
    
    # Train for a few epochs
    train_model(model, train_loader, device, epochs=5, optimizer=optimizer, loss_function=loss_function)
    
    # Evaluate finetuned model
    finetuned_accuracy = evaluate_model(model, test_loader, device)
    print(f"Accuracy after finetuning: {finetuned_accuracy:.4f} ({finetuned_accuracy*100:.2f}%)")
    print(f"Accuracy recovery: {(finetuned_accuracy - pruned_accuracy)*100:.2f}%")
    
    # Task 4.3: Save pruned model
    print("\nSaving pruned and finetuned model...")
    
    # Print model state dict keys before removing
    print("State dict keys before pruning removal:")
    print(model.state_dict().keys())
    
    # Remove the pruning parameters and convert to normal weight parameters
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.remove(module, 'weight')
    
    # Print model state dict keys after removing
    print("State dict keys after pruning removal:")
    print(model.state_dict().keys())
    
    # Save the model
    torch.save(model.state_dict(), 'mnist_model_pruned.pt')
    print("Pruned model saved as 'mnist_model_pruned.pt'")
    
    return baseline_accuracy, pruned_accuracy, finetuned_accuracy

# Main function for Task 4.2: Iterative pruning during training
def iterative_pruning():
    print("\n=== Task 4.2: Iterative pruning during training ===")
    
    # Load data
    train_loader, test_loader = load_mnist_data()
    
    # Select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a new model
    model = Network().to(device)
    
    # Define training parameters
    total_epochs = 20
    pruning_interval = 5  # Prune every 5 epochs
    pruning_steps = total_epochs // pruning_interval
    pruning_amount_per_step = 1 - (1 - 0.6) ** (1 / pruning_steps)  # Calculate amount per step for 60% total
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print(f"Training for {total_epochs} epochs with pruning every {pruning_interval} epochs")
    print(f"Pruning amount per step: {pruning_amount_per_step:.4f}")
    
    # Parameters to prune
    parameters_to_prune = (
        (model.fc1, 'weight'),
        (model.fc2, 'weight'),
        (model.fc3, 'weight'),
    )
    
    # Train with iterative pruning
    all_losses = []
    accuracies = []
    sparsities = []
    
    # Define function to calculate model sparsity
    def calculate_model_sparsity(model):
        total_weights = 0
        zero_weights = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                total_weights += param.numel()
                zero_weights += torch.sum(param == 0).item()
        return zero_weights / total_weights if total_weights > 0 else 0
    
    # Initial evaluation
    initial_accuracy = evaluate_model(model, test_loader, device)
    accuracies.append(initial_accuracy)
    sparsities.append(0.0)
    
    # Training and pruning loop
    for epoch in range(total_epochs):
        # Train for one epoch
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
        
        avg_loss = running_loss / batch_count
        all_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{total_epochs}, Loss: {avg_loss:.4f}")
        
        # Prune after every pruning_interval epochs
        if (epoch + 1) % pruning_interval == 0:
            print(f"Pruning at epoch {epoch+1}...")
            
            # Prune model
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_amount_per_step,
            )
            
            # Calculate current sparsity
            current_sparsity = calculate_model_sparsity(model)
            sparsities.append(current_sparsity)
            print(f"Current sparsity: {current_sparsity:.4f} ({current_sparsity*100:.2f}%)")
            
            # Evaluate model
            current_accuracy = evaluate_model(model, test_loader, device)
            accuracies.append(current_accuracy)
            print(f"Current accuracy: {current_accuracy:.4f} ({current_accuracy*100:.2f}%)")
    
    # Final evaluation
    final_accuracy = evaluate_model(model, test_loader, device)
    print(f"\nFinal accuracy after iterative pruning: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"Final sparsity: {calculate_model_sparsity(model):.4f} ({calculate_model_sparsity(model)*100:.2f}%)")
    
    # Remove pruning parameters
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.remove(module, 'weight')
    
    # Save the model
    torch.save(model.state_dict(), 'mnist_model_iterative_pruned.pt')
    print("Iteratively pruned model saved as 'mnist_model_iterative_pruned.pt'")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(all_losses) + 1), all_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True)
    plt.savefig('iterative_pruning_loss.png')
    
    # Plot accuracy vs sparsity
    plt.figure(figsize=(10, 6))
    plt.plot(sparsities, accuracies, 'o-')
    plt.xlabel('Sparsity')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Sparsity')
    plt.grid(True)
    plt.savefig('accuracy_vs_sparsity.png')
    
    return initial_accuracy, final_accuracy

# Main execution
if __name__ == "__main__":
    print("Neural Network Pruning for MNIST Classification")
    
    # Execute Task 4.1
    baseline_acc, pruned_acc, finetuned_acc = prune_after_training()
    
    # Execute Task 4.2
    initial_acc, iterative_pruned_acc = iterative_pruning()
    
    # Compare results
    print("\n=== Results Comparison ===")
    print(f"Baseline accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print(f"One-shot pruned accuracy: {pruned_acc:.4f} ({pruned_acc*100:.2f}%)")
    print(f"Finetuned after pruning accuracy: {finetuned_acc:.4f} ({finetuned_acc*100:.2f}%)")
    print(f"Iterative pruning accuracy: {iterative_pruned_acc:.4f} ({iterative_pruned_acc*100:.2f}%)")
    
    if iterative_pruned_acc > finetuned_acc:
        print("\nIterative pruning achieved better accuracy than one-shot pruning with finetuning.")
    else:
        print("\nOne-shot pruning with finetuning achieved better accuracy than iterative pruning.")