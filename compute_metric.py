import torch
import os
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image.inception import InceptionScore
import cleanfid
from tqdm import tqdm

# === CONFIG ===
image_folder = "generated_images_epoch_100"
batch_size = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Custom Dataset Loader ===
class GeneratedImagesDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = sorted([
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith((".png", ".jpg", ".jpeg"))
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
print("🧠 Computing Inception Score...")
is_metric = InceptionScore(normalize=True).to(device)
for batch in tqdm(dataloader, desc="IS Batches"):
    batch = batch.to(device)
    is_metric.update(batch)
inception_score, inception_std = is_metric.compute()
print(f"✅ Inception Score: {inception_score:.2f} ± {inception_std:.2f}")

# === FID Computation ===
print("\n🧠 Computing FID...")
fid_score = cleanfid.compute_fid(image_folder, dataset_name="cifar10", dataset_split="test")
print(f"✅ FID Score (vs CIFAR-10 test): {fid_score:.2f}")
