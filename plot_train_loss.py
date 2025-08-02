import os
import torch
import matplotlib.pyplot as plt

checkpoint_dir = "checkpoints"
checkpoint_path = os.path.join(checkpoint_dir, "ddpm_model.pth")

device = torch.device("cpu")  # Force CPU for inference
checkpoint = torch.load(checkpoint_path, map_location=device)

track_train_loss = checkpoint['loss']

plt.semilogy(track_train_loss)
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.show()