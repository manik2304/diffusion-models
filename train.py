import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Adam
from einops import rearrange
import tqdm
import os
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from model import EMA, Unet, NoiseScheduler, AddGaussianNoise
from cifar10_processing import load_cifar10, visualize_cifar10

import wandb

# Define your experiment hyperparameters here
params = {
    "learning_rate": 2e-4,
    "batch_size": 128,
    "weight_decay": 1e-2,
    "num_warmup_steps": 200,
    "optimizer": "Adam",
    "lr_scheduler": "cosine_with_warmup",
}


# wandb.init(project="ddpm-hyperparam-tuning", config=params)
# wandb.run.name = f"""bs={params['batch_size']}-lr={params['learning_rate']:.0e}
# -warmup={params['num_warmup_steps']}--optim={params['optimizer']}"""

wandb.init(project="ddpm-training-cifar10", config=params)  # Initialize wandb for logging
wandb.run.name = f"""bs={wandb.config.batch_size}-lr={wandb.config.learning_rate:.0e}
-warmup={wandb.config.num_warmup_steps}--optim={wandb.config.optimizer}"""


## ----- Training Configuration ----- ##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")  # For testing purposes, we can use CPU
num_epoch = 500
scheduled_epoch = 500
print(f"Using device: {device}")
batch_size = wandb.config.batch_size  # Use batch size from wandb config
num_diffusion_timesteps = 1000 # Number of diffusion steps, standard for DDPM

## ----- Model Checkpoint Configuration ----- ##
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "ddpm_model.pth")
resume_training = True  # Set to True if you want to resume from a checkpoint
save_model = True  # Set to True if you want to save the model after training
hyperparam_tuning = False  # Set to True if you are tuning hyperparameters

## ----- Load Data ----- ##
train_dataloader = load_cifar10(batch_size=batch_size)  # Load CIFAR-10 dataset
print(f"Number of batches in train_dataloader: {len(train_dataloader)}")  # Print number of batches
 
## ----- Initialize Model, Optimizer, and Scheduler ----- ##
model = Unet().to(device)  # Initialize the Unet model
ema_model = EMA(model, decay=0.99)  # Initialize EMA for model weights
ema_model.to(device)  # Move EMA model to the device
noise_scheduler = NoiseScheduler(num_diffusion_timesteps=num_diffusion_timesteps, device=device)  # Initialize noise scheduler
add_noise = AddGaussianNoise()  # Initialize noise addition module
loss_fn = F.mse_loss  # Mean Squared Error loss function

## Initialize optimizer and learning rate scheduler
lr = params["learning_rate"]  # Use learning rate from wandb config
weight_decay = params["weight_decay"]  # Use weight decay from wandb config
num_warmup_steps = params["num_warmup_steps"]  # Use number of warmup steps from wandb config
num_training_steps = scheduled_epoch * len(train_dataloader)  # Total number of training steps
if params['optimizer'] == "AdamW":
    print("Using AdamW optimizer")
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # AdamW optimizer
elif params['optimizer'] == "Adam":
    print("Using Adam optimizer")
    optimizer = Adam(model.parameters(), lr=lr, eps=1e-8)  # Adam optimizer
else:
    raise ValueError("No optimizer.")



# cosine scheduler with warmup
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                               num_warmup_steps=num_warmup_steps, 
                                               num_training_steps=num_training_steps)
# linear scheduler
#lr_scheduler = get_linear_schedule_with_warmup(optimizer,
#                                               num_warmup_steps=num_warmup_steps, 
#                                               num_training_steps=num_training_steps)

## ----- Load Checkpoint (if exists and resume_training is True) ----- ##
start_epoch = 0
loss_history_path = os.path.join(checkpoint_dir, "training_loss.pth")

if resume_training and os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    prev_epoch_loss = checkpoint['epoch_loss'] # epoch loss from the checkpoint
    track_train_loss = checkpoint['loss']  # Load training loss history from checkpoint if available
    
    # Load training loss history from separate file if it exists
    if os.path.exists(loss_history_path):
        loss_history = torch.load(loss_history_path, map_location=device)
        track_train_loss = loss_history['loss']
        print(f"Loaded training loss history with {len(track_train_loss)} steps")
    else:
        track_train_loss = checkpoint.get('loss', [])  # Fallback to checkpoint loss if available
    
    print(f"Resumed training from epoch {start_epoch}")
else:
    track_train_loss = []  # Initialize list to track training loss
    print("Starting training from scratch")
    prev_epoch_loss = 10 # Initialize previous epoch loss to a high value
    

## ----- Training Loop ----- ##

print("Starting training.")
if hyperparam_tuning:
    print("This is a hyperparameter tuning run. Results will be logged to wandb.")
else:
    print("This is a standard training run. Results will be saved to the checkpoint file. Also wandb will log the training loss and learning rate.")

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
        ema_model.update(model)  # Update EMA model weights
        
        # Update progress bar with loss
        progress_bar.set_postfix({"loss": loss.item()})
        loss_list.append(loss.item())
        track_train_loss.append(loss.item())  # Track training loss against each training step

    epoch_loss = sum(loss_list) / len(loss_list)  # Calculate average loss for the epoch
    print(f"Epoch {epoch+1}/{num_epoch}, Average Loss: {epoch_loss:.4f}")  # Print average loss for the epoch
    
    # Log epoch, training loss, and learning rate to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": epoch_loss,
        "lr": optimizer.param_groups[0]["lr"]})
    ## ----- Save Model Checkpoint ----- ##
    
    # Always save training loss history
    loss_history_path = os.path.join(checkpoint_dir, "training_loss.pth")
    torch.save({
        'epoch': epoch,
        'loss': track_train_loss,
        'epoch_loss': epoch_loss
    }, loss_history_path)
        
    if save_model == True and epoch_loss < prev_epoch_loss:
        print(f"Saving model checkpoint at epoch {epoch + 1} with loss {epoch_loss:.4f}")
        prev_epoch_loss = epoch_loss  # Update previous epoch loss to current
        # Save the model checkpoint with complete training history
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'ema_model_state_dict': ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'epoch_loss': epoch_loss,
            'loss': track_train_loss,  # Always include complete training history
            'num_diffusion_timesteps': num_diffusion_timesteps,
            'batch_size': batch_size
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

print(f"Training completed for {epoch + 1} epochs.")
wandb.finish()
