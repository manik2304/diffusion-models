import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import torchvision
import matplotlib.pyplot as plt
from einops import rearrange
from tqdm import tqdm
import os
import time
from datetime import datetime
from transformers import get_cosine_schedule_with_warmup 

from model import Unet, NoiseScheduler, AddGaussianNoise, EMA
from cifar10_processing import load_cifar10, visualize_cifar10

import json

## ----- Inference Configuration ----- ##
checkpoint_path = "checkpoints/ddpm_model.pth"
total_samples = 20000  # Total samples you want to reach
batch_size = 64  # Batch size for sampling
image_size = 32 # it is 32x32 for CIFAR-10
num_timesteps = 1000  # Number of diffusion steps, standard for DDPM
save_dir = "generated_images_ddpm_sampling_1000_timesteps"
timing_filename = "ddpm_timing_results.json"


# === LOAD MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
checkpoint = torch.load(checkpoint_path, map_location=device)
epoch = checkpoint.get('epoch', 'unknown')

model = Unet().to(device)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()
ema_model = EMA(model, decay=0.99)  # Initialize EMA for model weights
ema_model.load_state_dict(checkpoint['ema_model_state_dict']) # We use EMA model for sampling
ema_model.ema_model.eval()  # Set EMA model to evaluation mode
ema_model.to(device)  # Ensure EMA model is on the correct device

noise_scheduler = NoiseScheduler(num_diffusion_timesteps=num_timesteps, device=device)

# === OUTPUT DIR AND RESUME LOGIC ===
os.makedirs(save_dir, exist_ok=True)

# Check existing images and load previous timing results
existing_images = [f for f in os.listdir(save_dir) if f.endswith('.png')]
samples_already_generated = len(existing_images)

# Load previous timing results if they exist
previous_total_time = 0
if os.path.exists(timing_filename):
    with open(timing_filename, 'r') as f:
        previous_results = json.load(f)
        previous_total_time = previous_results.get('total_time', 0)
        print(f"Previous results found: {previous_results.get('total_samples_generated', 0)} samples in {previous_total_time:.2f} seconds")

print(f"Found {samples_already_generated} existing images")
print(f"Target: {total_samples} total samples")

if samples_already_generated >= total_samples:
    print("Target already reached! No new samples to generate.")
    exit()

samples_to_generate = total_samples - samples_already_generated
print(f"Will generate {samples_to_generate} new samples")

# === SAMPLING LOOP ===
num_batches = samples_to_generate // batch_size
remaining_samples = samples_to_generate % batch_size
sample_idx = samples_already_generated  # Start from where we left off

print(f"Starting DDPM sampling for {samples_to_generate} new images...")
print(f"Will generate {num_batches} full batches of {batch_size} samples each")
if remaining_samples > 0:
    print(f"Plus 1 final batch of {remaining_samples} samples")
start_time = time.time()

with torch.no_grad():
    # Generate full batches
    for _ in tqdm(range(num_batches), desc="Generating full batches"):
        gen_image = torch.randn(batch_size, 3, image_size, image_size, device=device)

        for t in reversed(range(num_timesteps)):
            z = torch.randn_like(gen_image) if t > 0 else torch.zeros_like(gen_image)
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

            noise_pred = ema_model.ema_model(gen_image, t_tensor)  # Use ema_model.ema_model to access the actual model

            beta = rearrange(noise_scheduler.get_beta(t_tensor), 'b -> b 1 1 1')
            alpha = rearrange(noise_scheduler.get_alpha(t_tensor), 'b -> b 1 1 1')
            alpha_bar = rearrange(noise_scheduler.get_alpha_bar(t_tensor), 'b -> b 1 1 1')
            coeff_noise = (1 - alpha) / torch.sqrt(1 - alpha_bar)
            coeff = 1 / torch.sqrt(alpha)

            gen_image = coeff * (gen_image - coeff_noise * noise_pred) + torch.sqrt(beta) * z

        # Clamp and normalize to [0,1]
        gen_image = torch.clamp(gen_image, -1, 1)
        gen_image = (gen_image + 1) / 2

        # Save each image
        for i in range(batch_size):
            filename = os.path.join(save_dir, f"sample_{sample_idx:05d}.png")
            torchvision.utils.save_image(gen_image[i], filename)
            sample_idx += 1

    # Generate remaining samples if any
    if remaining_samples > 0:
        print(f"Generating final batch of {remaining_samples} samples...")
        gen_image = torch.randn(remaining_samples, 3, image_size, image_size, device=device)

        for t in reversed(range(num_timesteps)):
            z = torch.randn_like(gen_image) if t > 0 else torch.zeros_like(gen_image)
            t_tensor = torch.full((remaining_samples,), t, device=device, dtype=torch.long)

            noise_pred = ema_model.ema_model(gen_image, t_tensor)

            beta = rearrange(noise_scheduler.get_beta(t_tensor), 'b -> b 1 1 1')
            alpha = rearrange(noise_scheduler.get_alpha(t_tensor), 'b -> b 1 1 1')
            alpha_bar = rearrange(noise_scheduler.get_alpha_bar(t_tensor), 'b -> b 1 1 1')
            coeff_noise = (1 - alpha) / torch.sqrt(1 - alpha_bar)
            coeff = 1 / torch.sqrt(alpha)

            gen_image = coeff * (gen_image - coeff_noise * noise_pred) + torch.sqrt(beta) * z

        # Clamp and normalize to [0,1]
        gen_image = torch.clamp(gen_image, -1, 1)
        gen_image = (gen_image + 1) / 2

        # Save each image
        for i in range(remaining_samples):
            filename = os.path.join(save_dir, f"sample_{sample_idx:05d}.png")
            torchvision.utils.save_image(gen_image[i], filename)
            sample_idx += 1

# === TIMING RESULTS ===
end_time = time.time()
current_session_time = end_time - start_time
total_time = previous_total_time + current_session_time
total_samples_generated = sample_idx  # Final count of all samples
time_per_image = total_time / total_samples_generated
images_per_second = total_samples_generated / total_time

print(f"\nDDPM Sampling Results:")
print(f"New samples generated this session: {samples_to_generate}")
print(f"Time for this session: {current_session_time:.2f} seconds ({current_session_time/60:.2f} minutes)")
print(f"Total samples generated: {total_samples_generated}")
print(f"Total time (all sessions): {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"Time per image: {time_per_image:.4f} seconds")
print(f"Images per second: {images_per_second:.2f}")
print(f"Images saved to: {save_dir}")

# Save simplified timing results
timing_results = {
    "total_samples_generated": total_samples_generated,
    "total_time": total_time,
    "time_per_image": time_per_image,
    "images_per_sec": images_per_second
}
print(f"\nTiming Results:")
print(timing_results)

# Save timing results to file
with open(timing_filename, 'w') as f:
    json.dump(timing_results, f, indent=2)
print(f"Timing results saved to: {timing_filename}")



