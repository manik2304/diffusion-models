import torch
import os
import time
from datetime import datetime
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image.inception import InceptionScore
from cleanfid import fid
from tqdm import tqdm
import json

# === CONFIG ===
image_folder = "generated_images_ddpm_sampling_1000_timesteps"
batch_size = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Custom Dataset Loader ===
class GeneratedImagesDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = sorted([
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith((".png", ".jpg", ".jpeg")) # Filter for image files
        ])
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),  # Required by Inception V3
            transforms.ConvertImageDtype(torch.float),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = read_image(self.image_paths[idx]) / 255.0  # to [0,1]
        return self.transform(img)

# === Load Dataset ===
dataset = GeneratedImagesDataset(image_folder)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# === Inception Score Computation ===
print("Computing Inception Score...")
is_metric = InceptionScore(normalize=True).to(device)
for batch in tqdm(dataloader, desc="IS Batches"): # IS: Inception Score
    batch = batch.to(device)
    is_metric.update(batch)
inception_score, inception_std = is_metric.compute()
print(f"Inception Score: {inception_score:.2f} ± {inception_std:.2f}")

# === FID Computation ===
print("\nComputing FID...")
fid_score_train = fid.compute_fid(image_folder, dataset_name="cifar10", 
dataset_res=32,  mode="clean", dataset_split="train")

fid_score_test = fid.compute_fid(image_folder, dataset_name="cifar10", 
dataset_res=32,  mode="clean", dataset_split="test")
print(f"FID Score (Train): {fid_score_train:.2f}")
print(f"FID Score (Test): {fid_score_test:.2f}")


# === Save Results in Dictionary ===
results = {
    "sampling_method": "ddpm",
    "num_images": len(dataset),
    "inception_score": float(inception_score),
    "inception_std": float(inception_std),
    "fid_score_train": float(fid_score_train),
    "fid_score_test": float(fid_score_test)
}


print(f"\nResults Dictionary:")
print(results)

# Save results to file
results_filename = f"ddpm_metric_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(results_filename, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Metric results saved to: {results_filename}")
