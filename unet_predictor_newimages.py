import os
import cv2
import torch
import numpy as np
from torch import nn
from torchvision import transforms

# ================= CONFIGURATION =================
MODEL_PATH = "lesion_unet.pth" 
INPUT_DIR = "/deltos/c/maize/image_processing/data/leaf/22r/aleph/18.7"
OUTPUT_DIR = "unet_direct_results"

COLOR_PINK = (255, 100, 255)

# ================= MODEL DEFINITION (MATCHING TRAINING) =================
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Helper: Double Convolution (Conv -> BN -> ReLU -> Conv -> BN -> ReLU)
        def cbr(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                # The missing layers were here:
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

# ================= MAIN =================
def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found! You must run 'train_segmentor.py' first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading UNet from {MODEL_PATH}...")
    
    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))])
    
    print(f"Segmenting {len(files)} images...")
    
    # Transform: UNet expects 256x256 input
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    for i, fname in enumerate(files):
        path = os.path.join(INPUT_DIR, fname)
        img_bgr = cv2.imread(path)
        if img_bgr is None: continue
        
        h_orig, w_orig = img_bgr.shape[:2]
        
        # 1. Prepare Input
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = tf(img_rgb).unsqueeze(0).to(device)
        
        # 2. Inference
        with torch.no_grad():
            output = model(input_tensor)
            pred_mask_small = torch.sigmoid(output).cpu().numpy()[0, 0]
        
        # 3. Resize Prediction back to Original Size
        pred_mask = cv2.resize(pred_mask_small, (w_orig, h_orig))
        
        # 4. Threshold
        binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
        
        # 5. Save Overlay
        overlay = img_bgr.copy()
        cnts, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, COLOR_PINK, 2)
        
        base = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_unet_vis.jpg"), overlay)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_unet_mask.png"), binary_mask)
        
        if i % 10 == 0: print(f"[{i+1}/{len(files)}] {fname}")

    print(f"Done! Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()