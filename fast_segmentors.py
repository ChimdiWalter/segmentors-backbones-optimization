import os
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tifffile as tiff

# ================= CONFIGURATION =================
CSV_PATH = "/deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones/cpu_output_2/opt_summary.csv" # UPDATE THIS
IMG_DIR = "/deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones/leaves"                            # UPDATE THIS
MODEL_SAVE_PATH = "lesion_unet.pth"
BATCH_SIZE = 8
EPOCHS = 40
LR = 1e-4

class LesionSegmentationDataset(Dataset):
    def __init__(self, csv_path, img_dir):
        self.df = pd.read_csv(csv_path)
        # Filter for good examples only
        self.df = self.df[(self.df["status"] == "ok") & (self.df["score"] > 20)].reset_index(drop=True)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Load Image
        img_path = os.path.join(self.img_dir, row["filename"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1)) # HWC -> CHW

        # 2. Load Mask (Produced by Navier Script)
        mask_path = row["mask16_path"]
        mask = tiff.imread(mask_path)
        # Convert 16-bit (0-65535) to Binary (0.0-1.0)
        mask = (mask > 0).astype(np.float32)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, axis=0) # Add channel dim

        return torch.from_numpy(img), torch.from_numpy(mask)

# Simple UNet Implementation
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def cbr(in_c, out_c):
            return nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
        
        self.e1 = cbr(3, 32)
        self.e2 = cbr(32, 64)
        self.e3 = cbr(64, 128)
        self.pool = nn.MaxPool2d(2)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2 = cbr(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1 = cbr(64, 32)
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        
        x = self.d2(torch.cat([self.up2(x3), x2], dim=1))
        x = self.d1(torch.cat([self.up1(x), x1], dim=1))
        return self.final(x)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LesionSegmentationDataset(CSV_PATH, IMG_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    print("Training UNet Segmentor...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()