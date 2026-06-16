
# Code CNN Mohammad et Adham
# Projet : Classification d'images Fashion-MNIST avec un CNN léger
# Version enrichie avec des commentaires explicatifs détaillés
import os
import time
import random
import math
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Mode non interactif pour sauvegarder les figures sans les afficher
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
print("optimize RMS ")
# Réglages généraux


# Fonction pour fixer les seeds aléatoires (important pour la reproductibilité)
# Cela garantit que les résultats seront les mêmes à chaque exécution.
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Définition des chemins de travail :
# ROOT = dossier parent du fichier courant (permet de rester portable)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# Liste des noms de classes du dataset Fashion-MNIST
CLASS_NAMES = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']
# Fonction pour choisir automatiquement le meilleur périphérique de calcul :
# - MPS : GPU Apple Silicon
# - CUDA : GPU Nvidia
# - CPU : si aucun GPU n’est disponible
def pick_device():
    try:
        if torch.backends.mps.is_available():  # GPU Apple
            return torch.device("mps")
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = pick_device()
print(f"Device utilisé : {DEVICE}")

# 1) Chargement des données



# Transformations baseline : tensor + normalisation simple
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Téléchargement automatique du dataset Fashion-MNIST (60 000 train + 10 000 test)
train_set = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transform_train)
test_set  = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=transform_test)

batch_size = 128
# Sur macOS avec MPS, il faut éviter num_workers>0 et pin_memory=True
# pour éviter des erreurs liées au backend Metal.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_set,  batch_size=512, shuffle=False,
                          num_workers=0, pin_memory=False)

## 2) Définition du modèle CNN
 # Partie convolutionnelle (extraction de caractéristiques)
 # Chaque bloc : plusieurs convolutions + ReLU + MaxPooling pour réduire la taille
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Entrée: 1x28x28
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # conv 3x3 : 1 canal d’entrée → 32 filtres
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), # deuxième couche convolutionnelle
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # reduction de taille (28x28 → 14x14)

            nn.Conv2d(32, 64, 3, padding=1), # plus de filtres = plus de profondeur
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # Nouvelle réduction (14x14 → 7x7)
        )
        # Partie fully-connected (classification finale)
        self.classifier = nn.Sequential(
            nn.Flatten(),# Aplatissement du tenseur 64x7x7 en un vecteur
            nn.Dropout(0.25),# Régularisation (empêche le surapprentissage)
            nn.Linear(64*7*7, 128),# Couche dense intermédiaire
            nn.ReLU(inplace=True),
            # Deuxième Dropout
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )# Sortie = 10 classes

    def forward(self, x):# Définition du passage avant (forward)
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN().to(DEVICE)

## 3) Entraînement et évaluation

criterion = nn.CrossEntropyLoss()      # fonction de perte : CrossEntropy # baseline: sans label smoothing

optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)#on a utiliser plussieur optimizateur pour cette etude 
# optimizateur adam est le meilleur avec 92,91% de precision en 12eme epoch 
# Fonction pour entraîner le modèle pendant une époque complète
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()# met le modèle en mode entraînement (active dropout)
    total, running_loss = 0, 0.0
    for images, targets in loader:
        # Envoi des batchs sur le GPU/CPU
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(images)# passe avant (prédiction brute)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step() # mise à jour des poids
        running_loss += loss.item() * images.size(0)
        total += images.size(0)
    return running_loss / total# moyenne de la perte sur l’ensemble de l’époque
# Fonction pour évaluer le modèle sur le jeu de test
def evaluate(model, loader, device):
    model.eval()# désactive dropout 
    all_preds, all_tgts = [], []
    with torch.no_grad(): # pas besoin de calculer les gradients
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_tgts.append(targets.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_tgts)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro") # moyenne du F1 par classe
    cm = confusion_matrix(y_true, y_pred)
    return acc, f1m, cm, y_true, y_pred
# 4) Fonctions de visualisation


# Sauvegarde d'une matrice de confusion en image PNG
def save_confusion_matrix(cm, path_png):
    plt.figure(figsize=(6.2,5.5))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Matrice de confusion (test)")
    plt.colorbar()
    ticks = np.arange(len(CLASS_NAMES))
    # Pour rester simple et lisible : indices 0–9
    plt.xticks(ticks, ticks, rotation=45)
    plt.yticks(ticks, ticks)
    plt.xlabel("Prédit")
    plt.ylabel("Vrai")
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()
# Sauvegarde de quelques exemples mal classés
def save_misclassified_images(raw_test_images, y_true, y_pred, path_png, max_samples=16):
    errs = np.where(y_true != y_pred)[0]
    if len(errs) == 0:
        return
    sel = np.random.choice(errs, size=min(max_samples, len(errs)), replace=False)
    cols = 4
    rows = math.ceil(len(sel)/cols)
    plt.figure(figsize=(cols*2.2, rows*2.2))
    for i, idx in enumerate(sel, 1):
        img = raw_test_images[idx].squeeze(0).numpy()  # [1,28,28] -> [28,28]
        plt.subplot(rows, cols, i)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f"V:{y_true[idx]} P:{y_pred[idx]}", fontsize=8)
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()

# 5) Boucle principale

n_params = sum(p.numel() for p in model.parameters())
print(f"Nombre total de paramètres : {n_params:,}")
def main():
    epochs = 12
    best_acc = 0.0# pour suivre la meilleure accuracy atteinte
    best_path = os.path.join(RESULTS_DIR, "cnn_fmnist_baseline_best.pt")
    metrics_csv = os.path.join(RESULTS_DIR, "metrics_baseline.csv")
    cm_png = os.path.join(RESULTS_DIR, "cm_baseline.png")
    errs_png = os.path.join(RESULTS_DIR, "misclassified_baseline.png")

    start = time.time()
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch","train_loss","test_accuracy","test_f1_macro"])
        for ep in range(1, epochs+1): # Entraînement sur une époque complète
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            acc, f1m, cm, y_true, y_pred = evaluate(model, test_loader, DEVICE) # Évaluation sur les données de test
            writer.writerow([ep, f"{train_loss:.6f}", f"{acc:.6f}", f"{f1m:.6f}"]) # Écriture des résultats dans le fichier CSV
            print(f"Epoch {ep:02d}/{epochs} | loss={train_loss:.4f} | acc={acc:.4f} | f1m={f1m:.4f}")
# Si le modèle actuel est meilleur, on le sauvegarde
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), best_path)
                save_confusion_matrix(cm, cm_png)

    elapsed = (time.time()-start)/60.0
    print(f"Terminé en {elapsed:.1f} min | Meilleure accuracy={best_acc:.4f}")
    print(f"Meilleur modèle sauvegardé: {best_path}")
    print(f"Métriques CSV: {metrics_csv}")
    print(f"Matrice de confusion PNG: {cm_png}")

    # Sauvegarde d'exemples d'erreurs typiques
    raw_test_images = test_set.data.unsqueeze(1).float()/255.0  # [N,1,28,28]
    acc, f1m, cm, y_true, y_pred = evaluate(model, test_loader, DEVICE)
    save_misclassified_images(raw_test_images, y_true, y_pred, errs_png, max_samples=16)
    print(f"Exemples de mauvaises prédictions: {errs_png}")

if __name__ == "__main__":
    main()
