import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.amp import autocast
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from src.config import (
    CHECKPOINTS_DIR, OUTPUTS_DIR, CLASSES,
    DEVICE, USE_AMP, NUM_CLASSES
)
from src.dataset import get_dataloaders
from src.model import build_model

def plot_confusion_matrix(cm, class_names, save_path):
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
    
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Normalized Confusion Matrix - VisionIQ Surveillance",
        ylabel="True Activity Label",
        xlabel="Predicted Activity Label"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Text annotations in each cell
    thresh = cm_normalized.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_pct = cm_normalized[i, j] * 100
            raw_cnt = cm[i, j]
            ax.text(j, i, f"{raw_cnt}\n({val_pct:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black",
                    fontsize=8)
            
    fig.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Model on Test Split")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(CHECKPOINTS_DIR, "best_model.pth"))
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}. Please train the model first.")
        
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    model_name = checkpoint.get('model_name', 'r2plus1d_18')
    
    model = build_model(model_name=model_name, num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    print("Loading test split DataLoader...")
    _, _, test_loader, _ = get_dataloaders(batch_size=args.batch_size)
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    print(f"Running evaluation on {len(test_loader.dataset)} test clips...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            with autocast('cuda', enabled=USE_AMP and DEVICE.type == 'cuda'):
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    print("\n" + "="*50)
    print("           TEST SET EVALUATION RESULTS           ")
    print("="*50)
    print(f"Accuracy    : {acc*100:.2f}%")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")
    print("="*50)
    
    report_dict = classification_report(
        all_targets, all_preds,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0
    )
    report_str = classification_report(
        all_targets, all_preds,
        target_names=CLASSES,
        zero_division=0
    )
    print(report_str)
    
    # Save Report to JSON
    report_path = os.path.join(OUTPUTS_DIR, "evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report_dict, f, indent=4)
    print(f"Saved classification metrics report to: {report_path}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=range(NUM_CLASSES))
    cm_plot_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, CLASSES, cm_plot_path)
    print(f"Saved confusion matrix plot to: {cm_plot_path}")

if __name__ == "__main__":
    main()
