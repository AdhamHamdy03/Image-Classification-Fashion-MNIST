# -*- coding: utf-8 -*-
import os, math, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torchvision import datasets, transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
BEST_PT = os.path.join(RESULTS_DIR, "cnn_fmnist_baseline_best.pt")

CLASS_NAMES = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32,32,3,padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64*7*7,128), nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        return self.classifier(self.features(x))

def pick_device():
    try:
        if torch.backends.mps.is_available(): return torch.device("mps")
    except Exception:
        pass
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def main():
    device = pick_device()
    print("Device:", device)
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(BEST_PT, map_location=device))
    model.eval()

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_set = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=tf)

    idx = np.random.choice(len(test_set), size=16, replace=False)
    imgs, gts, preds = [], [], []
    with torch.no_grad():
        for i in idx:
            x, y = test_set[i]
            p = model(x.unsqueeze(0).to(device)).argmax(1).item()
            imgs.append(x.squeeze(0).numpy()); gts.append(y); preds.append(p)

    # grille
    cols=4; rows=4
    plt.figure(figsize=(cols*2.3, rows*2.3))
    for k,(im,y,p) in enumerate(zip(imgs,gts,preds),1):
        plt.subplot(rows,cols,k)
        plt.imshow(im, cmap='gray'); plt.axis('off')
        ok = "✓" if y==p else "✗"
        plt.title(f"{ok}  V:{CLASS_NAMES[y]}\nP:{CLASS_NAMES[p]}", fontsize=8)
    out = os.path.join(RESULTS_DIR, "inference_grid.png")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print("Saved:", out)

if __name__ == "__main__":
    main()
