import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define the model architecture
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # Adjusted input size based on the output size of the convolutional layers
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten the output of the convolutional layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def adversarial_train(model, device, train_loader, optimizer, epsilon):
    model.train()  # Set the model to training mode
    correct = 0
    total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        # Generate adversarial examples
        perturbed_data, adv_output = fgsm_attack(model, data, target, epsilon)
        
        # Update model parameters
        optimizer.zero_grad()
        output = model(perturbed_data)
        
        # Calculate loss
        target_flat = target.flatten().long()  # Convert target to Long data type
        loss = F.cross_entropy(output, target_flat)  # Use the flattened target tensor
        
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target_flat).sum().item()
        
        # Print adversarial example and predictions
        if total % 100 == 0:  # Print every 100 iterations
            print(f"Iteration: {total}, Loss: {loss.item()}, Accuracy: {100 * correct / total}%")
            
            # Plot original and adversarial images
            original_img = to_pil(data[0].cpu())
            adversarial_img = to_pil(perturbed_data[0].cpu())

            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(original_img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            axes[1].imshow(adversarial_img)
            axes[1].set_title('Adversarial Image')
            axes[1].axis('off')
            plt.show()
    
    print('Accuracy on adversarial examples: %f %%' % (100 * correct / total))


def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('Accuracy on test set: %f %%' % (100 * correct / total))

import torch
import torch.nn.functional as F

def one_hot_encode(target, num_classes):
    """
    Convert target tensor to one-hot encoded format.

    Args:
        target: Target tensor containing class indices.
        num_classes: Number of classes in the classification problem.

    Returns:
        one_hot_target: One-hot encoded target tensor.
    """
    one_hot_target = F.one_hot(target, num_classes=num_classes)
    return one_hot_target.float()

import torch
import torch.nn.functional as F

def fgsm_attack(model, input, target, epsilon):
    """
    FGSM attack implementation.

    Args:
        model: PyTorch model to be attacked.
        input: Input image tensor.
        target: Target class label.
        epsilon: Epsilon value for perturbation magnitude.

    Returns:
        adv_input: Adversarial input tensor.
        adv_output: Model output on the adversarial example.
    """
    model.eval()  # Set model to evaluation mode

    # Detach and clone the input tensor (ensure on the same device as model)
    input = input.detach().clone().to(device)
    input.requires_grad_(True)  # Mark input for gradient calculation

    # Original prediction
    output = model(input)

    if torch.is_tensor(target):  # If target is provided as tensor
        if target.dim() > 1:  # If target is provided as class probabilities
            target = target.argmax(dim=1)  # Convert probabilities to indices
        # Ensure target is 1D tensor
        target = target.view(-1)
    else:
        raise ValueError("Target must be provided as a tensor.")

    loss = F.cross_entropy(output, target)

    # Backward pass to obtain the gradient
    model.zero_grad()
    loss.backward()

    # FGSM attack: perturb the input image with gradient sign
    perturbation = epsilon * input.grad.sign()
    adv_input = input + perturbation
    adv_input = torch.clamp(adv_input, min=0.0, max=1.0)  # Clamp to valid range (0.0 to 1.0)

    # Perform inference on the adversarial example
    adv_output = model(adv_input)

    return adv_input, adv_output


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = SimpleModel().to(device)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training parameters
    epochs = 10
    epsilon = 0.1  # Epsilon value for FGSM attack

    # Adversarial training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}:")
        adversarial_train(model, device, train_loader, optimizer, epsilon=epsilon)
        # Evaluate model
        evaluate(model, device, test_loader)

# Run the main function
if __name__ == "__main__":
    main()
