import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from einops import rearrange
import tqdm
import os
from transformers import get_cosine_schedule_with_warmup 

from model import Unet, NoiseScheduler, AddGaussianNoise
from cifar10_processing import load_cifar10, visualize_cifar10

## ----- Training Configuration ----- ##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # For testing purposes, we can use CPU
batch_size = 64  # Batch size for training
num_epoch = 50
num_diffusion_timesteps = 1000 # Number of diffusion steps, standard for DDPM

## ----- Model Checkpoint Configuration ----- ##
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "ddpm_model.pth")
resume_training = False  # Set to True if you want to resume from a checkpoint

## ----- Load Data ----- ##
train_dataloader = load_cifar10(batch_size=batch_size)  # Load CIFAR-10 dataset
num_training_steps = num_epoch * len(train_dataloader)  # Total number of training steps
print(f"Number of batches in train_dataloader: {len(train_dataloader)}")  # Print number of batches
 
## ----- Initialize Model, Optimizer, and Scheduler ----- ##
model = Unet().to(device)  # Initialize the Unet model
noise_scheduler = NoiseScheduler(num_diffusion_timesteps=num_diffusion_timesteps, device=device)  # Initialize noise scheduler
add_noise = AddGaussianNoise()  # Initialize noise addition module
loss_fn = F.mse_loss  # Mean Squared Error loss function
optimizer = AdamW(model.parameters(), lr = 1e-4, weight_decay = 1e-2) # AdamW optimizer with learning rate 1e-4 and weight decay 1e-2
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)

## ----- Load Checkpoint (if exists and resume_training is True) ----- ##
start_epoch = 0
if resume_training and os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    track_train_loss = checkpoint['loss']
    print(f"Resumed training from epoch {start_epoch}")
else:
    track_train_loss = []  # Initialize list to track training loss
    print("Starting training from scratch")
    

## ----- Training Loop ----- ##


for epoch in range(start_epoch, num_epoch):
    model.train()  # Set model to training mode
    progress_bar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epoch}", unit="batch")
    #iterative = iter(train_dataloader)
    loss_list = []   # Initialize loss list for the epoch

    for batch in progress_bar:  # Iterate over batches in the training dataloader
        images, labels = batch # Get images from the batch, ignore the labels
        x = images.to(device)  # Move images to the device (GPU or CPU)
        time_steps = torch.randint(0, num_diffusion_timesteps,
                                   (x.shape[0],), device = device)  # Randomly sample time steps for each image in the batch)
        
        noisy_x, noise = add_noise(x, time_steps, noise_scheduler) # Add noise to the images based on the sampled time steps
        predicted_noise = model(noisy_x, time_steps)
        # Check for NaNs in the predicted noise
        if torch.isnan(predicted_noise).any():
            print("NaN detected in predicted noise, skipping this batch.")
            continue
        loss = loss_fn(predicted_noise, noise, reduction='mean')  # Calculate the loss between predicted noise and actual noise
        optimizer.zero_grad()
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model parameters
        lr_scheduler.step()  # Update learning rate
        progress_bar.set_postfix({"loss": loss.item()})
        loss_list.append(loss.item())
        track_train_loss.append(loss.item())  # Track training loss against each training step

    epoch_loss = sum(loss_list) / len(loss_list)  # Calculate average loss for the epoch
    print(f"Epoch {epoch+1}/{num_epoch}, Average Loss: {epoch_loss:.4f}")  # Print average loss for the epoch
    
    ## ----- Save Model Checkpoint ----- ##
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'epoch_loss': epoch_loss,
        'loss': track_train_loss,
        'num_diffusion_timesteps': num_diffusion_timesteps,
        'batch_size': batch_size
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

print(f"Training completed for {epoch + 1} epochs.")




