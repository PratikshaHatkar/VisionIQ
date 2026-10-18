import os
import sys
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DEVICE, NUM_CLASSES, CLASSES,
    BATCH_SIZE, SPLITS_DIR, USE_AMP
)
from src.dataset import get_dataloaders
from src.model import build_model

def run_verification_tests():
    print("=" * 60)
    print("      VISIONIQ PIPELINE STAGE-BY-STAGE VERIFICATION     ")
    print("=" * 60)
    print(f"Executing on: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        torch.cuda.reset_peak_memory_stats(0)
    print("-" * 60)
    
    # -------------------------------------------------------------
    # Stage 1: Split Verification & Zero-Leakage Check
    # -------------------------------------------------------------
    print("[Stage 1/5] Testing Split Integrity & Zero-Leakage...")
    import pandas as pd
    train_df = pd.read_csv(os.path.join(SPLITS_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(SPLITS_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))
    
    train_parents = set(train_df['parent_video_id'].unique())
    val_parents = set(val_df['parent_video_id'].unique())
    test_parents = set(test_df['parent_video_id'].unique())
    
    leakage_train_val = train_parents.intersection(val_parents)
    leakage_train_test = train_parents.intersection(test_parents)
    leakage_val_test = val_parents.intersection(test_parents)
    
    assert len(leakage_train_val) == 0, f"Leakage Train-Val: {leakage_train_val}"
    assert len(leakage_train_test) == 0, f"Leakage Train-Test: {leakage_train_test}"
    assert len(leakage_val_test) == 0, f"Leakage Val-Test: {leakage_val_test}"
    print(f"  [PASS] Zero Data Leakage verified across {len(train_parents)+len(val_parents)+len(test_parents)} source recordings.")
    print(f"  Distribution: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)} clips.")
    
    # -------------------------------------------------------------
    # Stage 2: Data Loading & Tensor Validation
    # -------------------------------------------------------------
    print("\n[Stage 2/5] Testing Video DataLoader & Tensor Pipeline...")
    train_loader, val_loader, test_loader, _ = get_dataloaders(batch_size=4)
    inputs, targets = next(iter(train_loader))
    
    assert inputs.shape == (4, 3, 16, 112, 112), f"Unexpected input shape: {inputs.shape}"
    assert not torch.isnan(inputs).any(), "Input contains NaNs!"
    assert not torch.isinf(inputs).any(), "Input contains Infs!"
    assert len(targets) == 4, f"Unexpected target length: {len(targets)}"
    print(f"  [PASS] Batch tensor shape: {tuple(inputs.shape)} (B, C, T, H, W)")
    print(f"  [PASS] Target labels in batch: {targets.tolist()} ({[CLASSES[t] for t in targets]})")
    
    # -------------------------------------------------------------
    # Stage 3: Model Architecture & Forward Pass
    # -------------------------------------------------------------
    print("\n[Stage 3/5] Testing Model Initialization & GPU Forward Pass...")
    model = build_model("r2plus1d_18", num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(DEVICE)
    model.eval()
    
    inputs_gpu = inputs.to(DEVICE)
    with torch.no_grad():
        with autocast('cuda', enabled=USE_AMP and DEVICE.type == 'cuda'):
            logits = model(inputs_gpu)
            
    assert logits.shape == (4, NUM_CLASSES), f"Unexpected logits shape: {logits.shape}"
    print(f"  [PASS] Forward pass successful. Output shape: {tuple(logits.shape)}")
    
    # -------------------------------------------------------------
    # Stage 4: Backward Pass, Gradient Flow & VRAM Safety
    # -------------------------------------------------------------
    print("\n[Stage 4/5] Testing Backward Pass, AMP & Memory Headroom...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler('cuda', enabled=USE_AMP and DEVICE.type == 'cuda')
    criterion = nn.CrossEntropyLoss()
    targets_gpu = targets.to(DEVICE)
    
    optimizer.zero_grad()
    with autocast('cuda', enabled=USE_AMP and DEVICE.type == 'cuda'):
        preds = model(inputs_gpu)
        loss = criterion(preds, targets_gpu)
        
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    if DEVICE.type == 'cuda':
        peak_vram = torch.cuda.max_memory_allocated(0) / (1024**2)
        print(f"  [PASS] Backward pass succeeded. Loss: {loss.item():.4f}")
        print(f"  [PASS] Peak VRAM consumed: {peak_vram:.1f} MB (Well within 4096 MB capacity!)")
        assert peak_vram < 3800, "VRAM consumption exceeded safe safety buffer!"
    else:
        print(f"  [PASS] Backward pass succeeded on CPU. Loss: {loss.item():.4f}")
        
    # -------------------------------------------------------------
    # Stage 5: Micro-Batch Overfitting Test (Gradient Verification)
    # -------------------------------------------------------------
    print("\n[Stage 5/5] Testing Micro-Batch Overfitting Convergence...")
    # Overfit on single batch to prove mathematical convergence
    initial_loss = loss.item()
    for step in range(10):
        optimizer.zero_grad()
        with autocast('cuda', enabled=USE_AMP and DEVICE.type == 'cuda'):
            outputs = model(inputs_gpu)
            step_loss = criterion(outputs, targets_gpu)
        scaler.scale(step_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
    final_loss = step_loss.item()
    print(f"  Initial Loss: {initial_loss:.4f} -> Final Loss after 10 steps: {final_loss:.4f}")
    assert final_loss < initial_loss, "Model failed to minimize loss on micro-batch!"
    print("  [PASS] Convergence verified. Gradients flow correctly.")
    
    print("\n" + "=" * 60)
    print("   ALL 5 VERIFICATION STAGES PASSED SUCCESSFULLY!    ")
    print("=" * 60)

if __name__ == "__main__":
    run_verification_tests()
