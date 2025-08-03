from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt


def load_cifar10(batch_size = 128):

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), # Randomly flip the image horizontally
        transforms.ToTensor(), # Convert the image to a tensor [0,1]
        transforms.Lambda(lambda x: 2*x - 1) # Normalize the image to [-1,1]
    ])

    print("Loading CIFAR-10 dataset...")
    print("It will download the dataset if it is not already present.")
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader


def visualize_cifar10(train_dataloader):
    # Get a batch of images and labels
    data_iter = iter(train_dataloader)
    images, labels = next(data_iter)

    # Take only the first 16 images for visualization
    images = images[:16]
    images = (images + 1) / 2  # Convert images back to [0, 1] for visualization

    # Create a grid of images
    grid_img = torchvision.utils.make_grid(images, nrow=4)

    # Plot the grid of images
    plt.figure(figsize=(4, 4))
    plt.imshow(grid_img.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    plt.axis('off')  # Turn off axis
    plt.show()  # Display the grid of images    

if __name__ == "__main__":
    train_dataloader = load_cifar10() # Load CIFAR-10 dataset and return a DataLoader with batch_size = batch_size (default 128)
    visualize_cifar10(train_dataloader) # Visualize a batch of images from the CIFAR-10 dataset