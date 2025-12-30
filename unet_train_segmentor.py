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
# 1. Path to the CSV created by your optimizer
CSV_PATH = "/deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones/cpu_output_3/opt_summary.csv"

# 2. Path to the folder containing original leaf images
IMG_DIR = "/deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones/leaves"

# 3. Output file name
MODEL_SAVE_PATH = "lesion_unet.pth"

# Training Settings
BATCH_SIZE = 8   
EPOCHS = 50      
LR = 1e-4        

# ================= DATASET CLASS =================
class LesionSegmentationDataset(Dataset):
    def __init__(self, csv_path, img_dir):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found at: {csv_path}")
            
        df = pd.read_csv(csv_path)
        
        # Clean up column names
        df.columns = df.columns.str.strip()

        # Filter: Only use rows where the optimizer finished successfully
        if "status" in df.columns:
            df = df[df["status"] == "ok"]
        
        # Check if mask path exists in CSV
        if "mask16_path" not in df.columns:
            raise ValueError("Error: CSV is missing 'mask16_path' column.")
        
        # --- PATH FIXING LOGIC ---
        # The CSV contains paths from '/cluster/VAST/...', but we are on '/deltos/...'
        # We assume the mask files are in the same folder as the CSV file.
        csv_dir = os.path.dirname(csv_path)
        
        valid_rows = []
        for index, row in df.iterrows():
            # Get just the filename of the mask (e.g. "leaf_mask16.tif")
            mask_filename = os.path.basename(row["mask16_path"])
            
            # Construct the local path
            local_mask_path = os.path.join(csv_dir, mask_filename)
            
            # Check if it exists locally
            if os.path.exists(local_mask_path):
                # Update the row with the correct local path
                row["mask16_path"] = local_mask_path
                valid_rows.append(row)
            else:
                # Debug only for the first failure
                if len(valid_rows) == 0 and index == 0:
                    print(f"DEBUG: Failed to find mask at {local_mask_path}")

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        self.img_dir = img_dir
        
        print(f"[Dataset] Found {len(self.df)} valid image/mask pairs for training.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # --- 1. Load Original Image ---
        img_path = os.path.join(self.img_dir, row["filename"])
        img = cv2.imread(img_path) # Loads as BGR
        
        if img is None:
            # Return zeros if file corrupt (prevents crash)
            return torch.zeros(3, 256, 256), torch.zeros(1, 256, 256)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        # Normalize to 0.0 - 1.0
        img = img.astype(np.float32) / 255.0
        # Permute to (Channels, Height, Width) for PyTorch
        img = np.transpose(img, (2, 0, 1))

        # --- 2. Load Generated Mask ---
        mask_path = row["mask16_path"]
        mask = tiff.imread(mask_path)
        
        # Convert 16-bit to Binary Float
        mask = (mask > 0).astype(np.float32)
        
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, axis=0)

        return torch.from_numpy(img), torch.from_numpy(mask)

# ================= MODEL ARCHITECTURE (UNet) =================
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        def cbr(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        
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
        
        x = self.up2(x3)
        x = self.d2(torch.cat([x, x2], dim=1))
        
        x = self.up1(x)
        x = self.d1(torch.cat([x, x1], dim=1))
        
        return self.final(x)

# ================= TRAINING LOOP =================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on: {device}")

    dataset = LesionSegmentationDataset(CSV_PATH, IMG_DIR)
    
    if len(dataset) == 0:
        print("CRITICAL ERROR: No valid data found. Check your CSV path and image folder.")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.5f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n[SUCCESS] Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()