import os
import torch
import matplotlib.pyplot as plt

checkpoint_dir = "checkpoints"
checkpoint_path = os.path.join(checkpoint_dir, "ddpm_model.pth")

device = torch.device("cpu")  # Force CPU for inference
checkpoint = torch.load(checkpoint_path, map_location=device)

track_train_loss = checkpoint['loss']

plt.figure(figsize=(10, 6))
plt.semilogy(track_train_loss)
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.grid(True, alpha=0.3)

# Save the plot as an image file
plot_path = os.path.join(checkpoint_dir, "training_loss_plot.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Training loss plot saved to: {plot_path}")

# Also try to show (might work in some environments)
plt.show()