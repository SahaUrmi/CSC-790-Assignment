import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from scripts.data_provide import get_dataset
from models.RLinear import RLinearClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import os

SAVE_DIR = "/home/sahau24/csc790project/HAR/results"
os.makedirs(SAVE_DIR, exist_ok=True)

class Args:
    dataset_name = 'har70+'
    window_size = 128
    step_size = 64
    batch_size = 64

arg = Args()

# --- Dataset loading ---
train_loader, test_loader = get_dataset(arg)

# --- Compute class weights from training data ---
y_train_all = np.array([y.item() for _, y in train_loader.dataset])
unique_classes = np.unique(y_train_all)
class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train_all)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# --- Number of classes ---
num_classes = len(unique_classes)

# train_loader, test_loader = get_dataset(arg)

# num_classes = len(np.unique([y for _, y in train_loader.dataset]))  # or hardcode 7

def train_model(model, train_loader, test_loader, device, num_epochs=20):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))


    for epoch in range(num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1).cpu()
                all_preds.append(preds)
                all_labels.append(yb)

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        acc = (all_preds == all_labels).mean()
        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Epoch {epoch+1:02d} | Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")

        # Save Confusion Matrix and Report
        if epoch == num_epochs - 1:
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"))
            plt.close()

            # Save classification report
            report = classification_report(all_labels, all_preds, digits=4,zero_division=0)
            with open(os.path.join(SAVE_DIR, "classification_report.txt"), "w") as f:
                f.write(report)
            

            # Save predicted class distribution
            unique, counts = np.unique(all_preds, return_counts=True)
            with open(os.path.join(SAVE_DIR, "predicted_distribution.txt"), "w") as f:
                f.write("Predicted class distribution:\n")
                for label, count in zip(unique, counts):
                    f.write(f"Class {label}: {count} samples\n")

            # Save per-class accuracy plot
            per_class_acc = cm.diagonal() / cm.sum(axis=1)
            plt.figure(figsize=(8, 4))
            sns.barplot(x=list(range(num_classes)), y=per_class_acc)
            plt.ylabel("Per-Class Accuracy")
            plt.xlabel("Class Label")
            plt.title("Accuracy per Class")
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, "per_class_accuracy.png"))
            plt.close()

            # Save multi-class ROC curve
            all_probs = []
            with torch.no_grad():
                for xb, _ in test_loader:
                    xb = xb.to(device)
                    logits = model(xb)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()  # [B, K]
                    all_probs.append(probs)

            all_probs = np.concatenate(all_probs, axis=0)  # shape: [N, K]
            y_bin = label_binarize(all_labels, classes=list(range(num_classes)))  # [N, K]

            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(num_classes):
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], all_probs[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            plt.figure(figsize=(10, 8))
            for i in range(num_classes):
                plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Multi-Class ROC Curve (One-vs-Rest)')
            plt.legend(loc='lower right')
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, "roc_curve.png"))
            plt.close()



model = RLinearClassifier(input_dim=128, proj_dim=64, num_classes=num_classes)
train_model(model, train_loader, test_loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


