import torch
import random
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage, transforms
import torchvision.utils as utils
from torchvision.transforms import Normalize
import torch.nn.functional as F

# Define mean and standard deviation for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Define normalization transformation
normalize = Normalize(mean=mean, std=std)

def pgd_attack(model, input, target, epsilon, alpha, iterations):
    """
    PGD attack implementation.

    Args:
        model: PyTorch model to be attacked.
        input: Input image tensor.
        target: Target class label (for untargeted attacks, set to None).
        epsilon: Epsilon value for perturbation magnitude.
        alpha: Step size for each iteration.
        iterations: Number of iterations for the iterative attack.

    Returns:
        adv_input: Adversarial input tensor after multiple iterations.
    """
    # Initialize adversarial input as a clone of the original input
    adv_input = input.clone().detach().to(device)
    adv_input.requires_grad_(True)
    
    for _ in range(iterations):
        # Forward pass
        output = model(adv_input)

        # Calculate loss
        num_classes = output.shape[1]
        loss = F.cross_entropy(output, target.repeat_interleave(adv_input.size(0)))


        # Backward pass
        model.zero_grad()
        loss.backward()

        # PGD step: perturb the input image with gradient sign and clip to epsilon
        adv_input = adv_input + alpha * adv_input.grad.sign()
        perturbation = torch.clamp(adv_input - input, min=-epsilon, max=epsilon)
        adv_input = torch.clamp(input + perturbation, min=0, max=1).detach()

        # Ensure that the perturbed image is within epsilon ball around original image
        adv_input = torch.max(torch.min(adv_input, input + epsilon), input - epsilon)
        adv_input = torch.clamp(adv_input, 0.0, 1.0)

        # Detach and re-require gradient for next iteration
        adv_input.requires_grad_(True)

    return adv_input

def adversarial_training(model, train_loader, epsilon, alpha, iterations, num_epochs, learning_rate):
    """
    Adversarial training implementation.

    Args:
        model: PyTorch model to be trained.
        train_loader: DataLoader containing the training dataset.
        epsilon: Epsilon value for PGD attack.
        alpha: Step size for PGD attack.
        iterations: Number of iterations for PGD attack.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            adv_inputs = pgd_attack(model, inputs, targets, epsilon, alpha, iterations)
            all_inputs = torch.cat([inputs, adv_inputs], dim=0)
            all_targets = torch.cat([targets, targets], dim=0)
            optimizer.zero_grad()
            outputs = model(all_inputs)
            loss = criterion(outputs, all_targets)
            loss.backward()
            optimizer.step()

# Example usage after performing PGD attack
epsilon = 0.05  # Epsilon value for PGD attack
alpha = 0.01  # Step size for PGD attack
iterations = 10  # Number of iterations for PGD attack
num_epochs = 10  # Number of epochs for adversarial training
learning_rate = 0.001  # Learning rate for optimizer

# Perform adversarial training
adversarial_training(model, train_loader, epsilon, alpha, iterations, num_epochs, learning_rate)
