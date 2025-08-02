import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import torchvision
import matplotlib.pyplot as plt
from einops import rearrange
from tqdm import tqdm
import os
from transformers import get_cosine_schedule_with_warmup 

from model import Unet, NoiseScheduler, AddGaussianNoise
from cifar10_processing import load_cifar10, visualize_cifar10

## ----- Inference Configuration ----- ##
checkpoint_path = "checkpoints/ddpm_model.pth"
total_samples = 5000
batch_size = 500
image_size = 32 # it is 32x32 for CIFAR-10
num_timesteps = 1000  # Number of diffusion steps, standard for DDPM


# === LOAD MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
checkpoint = torch.load(checkpoint_path, map_location=device)
epoch = checkpoint.get('epoch', 'unknown')

model = Unet().to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

noise_scheduler = NoiseScheduler(num_diffusion_timesteps=num_timesteps, device=device)

# === OUTPUT DIR ===
save_dir = f"generated_images_epoch_{epoch+1}"
os.makedirs(save_dir, exist_ok=True)

# === SAMPLING LOOP ===
num_batches = total_samples // batch_size
sample_idx = 0


with torch.no_grad():
    for _ in tqdm(range(num_batches), desc="Generating samples"):
        gen_image = torch.randn(batch_size, 3, image_size, image_size, device=device)

        for t in reversed(range(num_timesteps)):
            z = torch.randn_like(gen_image) if t > 0 else torch.zeros_like(gen_image)
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

            noise_pred = model(gen_image, t_tensor)

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

