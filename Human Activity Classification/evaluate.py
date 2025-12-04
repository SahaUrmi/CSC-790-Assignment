from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import os


def evaluate_model_metrics(model, test_loader, arg, max_f1):
    """
    Evaluates model performance and prints accuracy, precision, recall, F1-score,
    confusion matrix, and plots ROC curve.

    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test set
        device: 'cuda' or 'cpu'
        num_classes: Number of output classes
    """
    if arg.dataset_name =="har70+":
        num_classes=7
        result_path="/home/sahau24/csc790project/HAR/results/mamba/"

    model.eval()
    model.to('cuda')
    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to('cuda'), yb.to('cuda')
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)  # get probabilities
            preds = torch.argmax(probs, dim=1)

            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probs.cpu().numpy())  # for ROC curve

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # Precision, Recall, F1 (macro average)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    if f1>max_f1:
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig(result_path+"confusion.png")

        # ROC Curve (multi-class)
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(8, 6))
        for i in range(num_classes):
            plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.4f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Multi-class)")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(result_path+"roc.png")

    return acc, precision, recall, f1