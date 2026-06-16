# -*- coding: utf-8 -*-
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
CSV = os.path.join(RES, "metrics_baseline.csv")

# lire métriques
epochs, loss, acc, f1 = [], [], [], []
with open(CSV, "r") as f:
    rd = csv.DictReader(f)
    for r in rd:
        epochs.append(int(r["epoch"]))
        loss.append(float(r["train_loss"]))
        acc.append(float(r["test_accuracy"]))
        f1.append(float(r["test_f1_macro"]))

# sauver résumé (ajoute une ligne)
summary_path = os.path.join(RES, "summary.csv")
best_idx = int(np.argmax(acc))
with open(summary_path, "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["experience","accuracy","f1_macro","epochs","comment"])
    wr.writerow(["baseline", f"{acc[best_idx]:.4f}", f"{f1[best_idx]:.4f}", epochs[best_idx],
                 "CNN léger (2xConv blocs)"])

# courbes
plt.figure(); plt.plot(epochs, loss, marker="o"); plt.xlabel("Époque"); plt.ylabel("Loss train")
plt.title("Évolution de la perte (train)"); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(RES, "curve_loss.png"), dpi=150); plt.close()

plt.figure(); plt.plot(epochs, acc, marker="o"); plt.xlabel("Époque"); plt.ylabel("Accuracy test")
plt.title("Accuracy vs époques"); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(RES, "curve_acc.png"), dpi=150); plt.close()

plt.figure(); plt.plot(epochs, f1, marker="o"); plt.xlabel("Époque"); plt.ylabel("Macro-F1 test")
plt.title("Macro-F1 vs époques"); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(RES, "curve_f1.png"), dpi=150); plt.close()

print("OK -> summary.csv, curve_loss.png, curve_acc.png, curve_f1.png")
