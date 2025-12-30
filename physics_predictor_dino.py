import os
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ================= CONFIGURATION =================
CSV_PATH = "/deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones/cpu_output_3/opt_summary.csv"
IMG_DIR = "/deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones/leaves"
MODEL_SAVE_PATH = "dino_param_predictor.pth"
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-3 # Higher LR because we are only training the head

# We predict 7 physics parameters
NUM_PARAMS = 7 

class PhysicsParamDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        if not os.path.exists(csv_path): raise FileNotFoundError(csv_path)
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        # Filter valid
        if "status" in df.columns: df = df[df["status"] == "ok"]
        # Use ALL data
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        print(f"[DINO Dataset] Training on {len(self.df)} images.")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        img = cv2.imread(img_path)
        if img is None: return torch.zeros(3, 224, 224), torch.zeros(NUM_PARAMS)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform: img = self.transform(img)

        params = np.array([
            float(row["mu"]), float(row["lambda"]), float(row["diffusion_rate"]), 
            float(row["alpha"]), float(row["beta"]), float(row["gamma"]), 
            float(row["energy_threshold"]) * 0.01 
        ], dtype=np.float32)
        return img, torch.from_numpy(params)

class DINOv2Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        # Load DINOv2 Small (good speed/accuracy balance)
        print("Loading DINOv2 from Torch Hub...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        # FREEZE the backbone (We don't have enough data to retrain DINO)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # DINOv2 Small output dimension is 384
        self.head = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_PARAMS)
        )

    def forward(self, x):
        # DINO expects images to be multiples of 14. 224 is perfect.
        # It returns raw features.
        features = self.backbone(x)
        return self.head(features)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # DINOv2 requires specific Normalization values
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)), # Must be multiple of 14
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = PhysicsParamDataset(CSV_PATH, IMG_DIR, transform=tf)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = DINOv2Regressor().to(device)
    # Only optimize the 'head' parameters, not the backbone
    optimizer = optim.Adam(model.head.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        # Ensure backbone stays in eval mode (for BatchNorm/Dropout consistency)
        model.backbone.eval() 
        
        running_loss = 0.0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(loader):.6f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[DONE] DINO Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()