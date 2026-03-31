import os
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ================= CONFIGURATION =================
CSV_PATH = "/deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones/cpu_output_3/opt_summary.csv"
IMG_DIR = "/deltos/e/lesion_phes/code/python/pipeline/segmentors_backbones/leaves"
MODEL_SAVE_PATH = "param_predictor_resnet.pth"
BATCH_SIZE = 16
EPOCHS = 25
LR = 1e-4
NUM_PARAMS = 7

class PhysicsParamDataset(Dataset):
    # CHANGE: Set default min_score_quantile to 0.0 to use ALL data
    def __init__(self, csv_path, img_dir, transform=None, min_score_quantile=0.0):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        
        # Strip whitespace from columns just in case
        df.columns = df.columns.str.strip()

        # 1. Check columns
        required_cols = ["mu", "lambda", "diffusion_rate", "alpha", "beta", "gamma", "energy_threshold"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CSV is missing column '{col}'.")

        # 2. Filter valid rows only (must be 'ok')
        if "status" in df.columns:
            df = df[(df["status"] == "ok")].copy()
        
        # 3. Filter by quality (0.0 means KEEP EVERYTHING)
        if "score" in df.columns and not df.empty:
            df = df[np.isfinite(pd.to_numeric(df["score"], errors='coerce'))]
            q = df["score"].quantile(min_score_quantile)
            df = df[df["score"] >= q].reset_index(drop=True)

        self.df = df
        self.img_dir = img_dir
        self.transform = transform

        print(f"[Dataset] Using {len(self.df)} examples (100% of successful runs).")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            # Dummy fallback
            return torch.zeros(3, 224, 224), torch.zeros(NUM_PARAMS)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)

        params = np.array([
            float(row["mu"]), float(row["lambda"]), float(row["diffusion_rate"]),
            float(row["alpha"]), float(row["beta"]), float(row["gamma"]),
            float(row["energy_threshold"]) * 0.01,
        ], dtype=np.float32)

        return img, torch.from_numpy(params)

def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_PARAMS)
    return model

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # CHANGE: Set quantile to 0.0 here explicitly
    dataset = PhysicsParamDataset(CSV_PATH, IMG_DIR, transform=tf, min_score_quantile=0.0)
    
    if len(dataset) == 0:
        print("Error: Dataset is empty.")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    model = get_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(loader))
        print(f"Epoch {epoch+1}/{EPOCHS} | MSE Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[DONE] Saved param model to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()