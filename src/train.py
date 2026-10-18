import os
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score

from src.config import (
    CHECKPOINTS_DIR, CLASSES, NUM_CLASSES,
    BATCH_SIZE, GRAD_ACCUM_STEPS, LEARNING_RATE,
    WEIGHT_DECAY, NUM_EPOCHS, PATIENCE,
    DEVICE, USE_AMP, RANDOM_SEED
)
from src.dataset import set_seed, get_dataloaders, compute_class_weights
from src.model import build_model

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch_idx=1, total_epochs=1, grad_accum_steps=GRAD_ACCUM_STEPS, debug=False):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx:02d}/{total_epochs:02d} [Train]", unit="batch", leave=False)
    
    for step, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        with autocast('cuda', enabled=USE_AMP and device.type == 'cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_scaled = loss / grad_accum_steps
            
        if scaler and device.type == 'cuda':
            scaler.scale(loss_scaled).backward()
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss_scaled.backward()
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())
        
        # Live progress metrics in tqdm
        current_loss = running_loss / len(all_targets)
        current_acc = accuracy_score(all_targets, all_preds)
        pbar.set_postfix({
            'loss': f"{current_loss:.3f}",
            'acc': f"{current_acc:.3f}"
        })
        
        if debug and step >= 3:
            break
            
    total_samples = len(all_targets)
    epoch_loss = running_loss / total_samples
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    return epoch_loss, epoch_acc, epoch_f1

def validate(model, dataloader, criterion, device, epoch_idx=1, total_epochs=1, debug=False):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx:02d}/{total_epochs:02d} [Val]  ", unit="batch", leave=False)
    
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            with autocast('cuda', enabled=USE_AMP and device.type == 'cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
            current_loss = running_loss / len(all_targets)
            current_acc = accuracy_score(all_targets, all_preds)
            pbar.set_postfix({
                'loss': f"{current_loss:.3f}",
                'acc': f"{current_acc:.3f}"
            })
            
            if debug and step >= 3:
                break
                
    total_samples = len(all_targets)
    val_loss = running_loss / total_samples
    val_acc = accuracy_score(all_targets, all_preds)
    val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    return val_loss, val_acc, val_f1

def plot_training_history(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='blue', marker='o')
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='red', marker='s')
    ax1.set_title('Cross Entropy Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, history['val_acc'], label='Val Accuracy', color='green', marker='o')
    ax2.plot(epochs, history['val_f1'], label='Val Macro F1', color='purple', marker='^')
    ax2.set_title('Validation Performance')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train Video Action Recognition Model")
    parser.add_argument("--model", type=str, default="r2plus1d_18", choices=["r2plus1d_18", "mobilenet_gru"])
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--freeze_early", action="store_true", default=True, help="Freeze early backbone layers for 60% faster training")
    parser.add_argument("--full_finetune", dest="freeze_early", action="store_false", help="Unfreeze all layers for full end-to-end training")
    parser.add_argument("--debug", action="store_true", help="Quick smoke test mode with few batches")
    args = parser.parse_args()
    
    set_seed(RANDOM_SEED)
    print(f"Using Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Initial VRAM Allocated: {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB")
        
    print(f"Loading data with batch size {args.batch_size}...")
    train_loader, val_loader, _, train_df = get_dataloaders(batch_size=args.batch_size)
    
    # Class weights to address imbalance
    class_weights = compute_class_weights(train_df).to(DEVICE)
    print("Class weights:", {CLASSES[i]: round(class_weights[i].item(), 2) for i in range(NUM_CLASSES)})
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    print(f"Initializing model: {args.model} (pretrained=True, freeze_early={args.freeze_early})...")
    model = build_model(model_name=args.model, num_classes=NUM_CLASSES, pretrained=True, freeze_early_layers=args.freeze_early)
    model = model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=USE_AMP and DEVICE.type == 'cuda')
    
    best_val_f1 = -1.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    log_csv_path = os.path.join(CHECKPOINTS_DIR, "training_log.csv")
    log_rows = []
    
    print(f"\n--- Starting Training ({args.epochs} epochs) ---")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE,
            epoch_idx=epoch, total_epochs=args.epochs, debug=args.debug
        )
        val_loss, val_acc, val_f1 = validate(
            model, val_loader, criterion, DEVICE,
            epoch_idx=epoch, total_epochs=args.epochs, debug=args.debug
        )
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} F1: {train_f1:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} F1: {val_f1:.3f}")
        sys.stdout.flush()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        log_rows.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'lr': optimizer.param_groups[0]['lr']
        })
        pd.DataFrame(log_rows).to_csv(log_csv_path, index=False)
        
        # Save Best Checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_path = os.path.join(CHECKPOINTS_DIR, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'model_name': args.model,
                'classes': CLASSES
            }, best_path)
            print(f"  --> Saved new best checkpoint to {best_path} (Val Macro F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE and not args.debug:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
                
    total_duration = time.time() - start_time
    print(f"\nTraining completed in {total_duration/60:.2f} minutes. Best Val Macro F1: {best_val_f1:.4f}")
    
    # Save last model checkpoint
    last_path = os.path.join(CHECKPOINTS_DIR, "last_model.pth")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'val_f1': val_f1,
        'model_name': args.model,
        'classes': CLASSES
    }, last_path)
    
    # Plot curve
    plot_training_history(history, os.path.join(CHECKPOINTS_DIR, "loss_curve.png"))
    print(f"Saved loss curve to {os.path.join(CHECKPOINTS_DIR, 'loss_curve.png')}")

if __name__ == "__main__":
    main()
