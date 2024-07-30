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
        # loss = torch.nn.functional.cross_entropy(output, torch.tensor([target]).to(device) if target is not None else output.detach().max(1)[1])
        num_classes = output.shape[1]
        # target_one_hot = F.one_hot(target, num_classes=num_classes)
        loss = F.cross_entropy(output, target)

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


def perform_pgd_attack_on_subset(model, data_loader, num_images, epsilon, alpha, iterations):
    """
    Perform PGD adversarial attack on a random subset of images from the dataset and visualize the results.

    Args:
        model: PyTorch model to be attacked.
        data_loader: DataLoader containing the dataset.
        num_images: Number of images to select randomly from the dataset.
        epsilon: Epsilon value for perturbation magnitude.
        alpha: Step size for each iteration.
        iterations: Number of iterations for the iterative attack.
    """
    images_to_attack = random.sample(range(len(data_loader.dataset)), num_images)
    for index in images_to_attack:
        input, target = data_loader.dataset[index]
        input, target = input.unsqueeze(0).to(device), torch.tensor(target).long().to(device) # Move to device and convert target to tensor

        # PGD attack
        adv_input = pgd_attack(model, input, target, epsilon=epsilon, alpha=alpha, iterations=iterations)

        # Perform inference on the adversarial example
        adv_output = model(adv_input)

        # Check attack status
        print('Image Index:', index)
        print('Attack Status:', 'Successful' if torch.argmax(adv_output).item() != target.item() else 'Unsuccessful')

        # Calculate adversarial prediction
        output_class_adversarial = torch.argmax(adv_output).item()
        print('Original Prediction:', target.item())
        print('Adversarial Prediction:', output_class_adversarial)
        print('------------------------')
        to_pil = transforms.ToPILImage()
        # Convert tensors to PIL images for visualization
        original_img = to_pil(input.squeeze(0).cpu())
        adversarial_img = to_pil(adv_input.squeeze(0).cpu())

        # Plot original and adversarial images
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        axes[1].imshow(adversarial_img)
        axes[1].set_title('Adversarial Image')
        axes[1].axis('off')
        plt.show()


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example usage
num_images_to_attack = 20  # Number of images to attack (reduced for demonstration)
epsilon_testing = 0.05  # Epsilon value for perturbation magnitude
alpha = 0.01  # Step size for each iteration
iterations = 10  # Number of iterations for PGD

print('Performing PGD adversarial attack on a subset of images...')
perform_pgd_attack_on_subset(model, test_loader, num_images_to_attack, epsilon_testing, alpha, iterations)
