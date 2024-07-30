import torch
import random
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage, transforms
import torchvision.utils as utils
to_pil = ToPILImage()

# Normalize input image
normalize = transforms.Normalize(mean=[.5], std=[.5])

# Define device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fgsm_attack(model, input, target, epsilon):
    """
    FGSM attack implementation.

    Args:
        model: PyTorch model to be attacked.
        input: Input image tensor.
        target: Target class label (for untargeted attacks, set to None).
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
    loss = torch.nn.functional.cross_entropy(output, torch.tensor([target]).to(device) if target is not None else output.detach().max(1)[1])

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


def iterative_fgsm_attack(model, input, target, epsilon, iterations):
    """
    Iterative FGSM attack implementation.

    Args:
        model: PyTorch model to be attacked.
        input: Input image tensor.
        target: Target class label (for untargeted attacks, set to None).
        epsilon: Epsilon value for perturbation magnitude.
        iterations: Number of iterations for the iterative attack.

    Returns:
        adv_input: Adversarial input tensor after multiple iterations.
    """
    for _ in range(iterations):
        adv_input, _ = fgsm_attack(model, input, target, epsilon)
        input = adv_input.clone().detach()  # Detach and clone for next iteration
        input.requires_grad_(True)  # Mark input for gradient calculation in the next iteration
    return adv_input


def perform_attack_on_subset(model, data_loader, num_images, epsilon, iterations):
    """
    Perform adversarial attack on a random subset of images from the dataset and visualize the results.

    Args:
        model: PyTorch model to be attacked.
        data_loader: DataLoader containing the dataset.
        num_images: Number of images to select randomly from the dataset.
        epsilon: Epsilon value for perturbation magnitude.
        iterations: Number of iterations for the iterative attack.
    """
    images_to_attack = random.sample(range(len(data_loader.dataset)), num_images)
    for index in images_to_attack:
        input, target = data_loader.dataset[index]
        input, target = input.unsqueeze(0).to(device), torch.tensor(target).to(device)  # Move to device and convert target to tensor

        # Normalize input
        input_normalized = normalize(input)

        # Choose attack function
        attack_fn = fgsm_attack if iterations == 0 else iterative_fgsm_attack

        # FGSM attack with conditional unpacking
        if iterations == 0:
            adv_input, adv_output = attack_fn(model, input_normalized, None, epsilon=epsilon)  # Untargeted attack (target=None)
        else:
            adv_input = attack_fn(model, input_normalized, None, epsilon=epsilon, iterations=iterations)  # Untargeted attack
            adv_output = model(adv_input)  # Obtain output for iterative FGSM

        # Check attack status
        print('Image Index:', index)
        print('Attack Status:', 'Successful' if torch.argmax(adv_output).item() != target.item() else 'Unsuccessful')

        # Calculate adversarial prediction
        output_class_adversarial = torch.argmax(adv_output).item()
        print('Original Prediction:', target.item())
        print('Adversarial Prediction:', output_class_adversarial)
        print('------------------------')

        # Convert tensors to PIL images for visualization
        original_img = to_pil(input_normalized.squeeze(0).cpu())
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

# Example usage
num_images_to_attack = 20  # Number of images to attack (reduced for demonstration)
epsilon_testing = 0.55  # Epsilon value for perturbation magnitude
iterations = 5  # Number of iterations for iterative FGSM

print('Performing adversarial attack on a subset of images...')
perform_attack_on_subset(model, test_loader, num_images_to_attack, epsilon_testing, iterations)
