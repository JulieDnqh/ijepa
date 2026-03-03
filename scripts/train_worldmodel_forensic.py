"""
World Model Forensic Training Pipeline v3 — Gated Fusion.

Trains ForensicHead using:
  1. Multi-scale encoder features (5120-dim, from frozen I-JEPA Encoder)
  2. Prediction error maps (1-dim, precomputed from finetuned I-JEPA Predictor)

Uses Gated Fusion architecture with per-epoch contribution logging.
Uses precomputed error maps from `error_cache/` directory for fast training.
Uses existing dataset_splits.json for 80/10/10 split.
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse

from src.models.multiscale_encoder import MultiScaleEncoder
from src.models.forensic_head import ForensicHead


class CASIAWorldModelDataset(Dataset):
    """
    CASIA 2.0 dataset with precomputed prediction error maps.
    Each sample returns: (image_tensor, error_map, gt_mask)
    """
    def __init__(self, casia_dir, file_list, error_cache_dir, transform=None):
        self.tp_dir = os.path.join(casia_dir, 'Tp')
        self.gt_dir = os.path.join(casia_dir, 'CASIA 2 Groundtruth')
        self.error_cache_dir = error_cache_dir
        self.transform = transform
        self.samples = []
        
        for f in file_list:
            basename = os.path.splitext(f)[0]
            gt_path = os.path.join(self.gt_dir, basename + "_gt.png")
            err_path = os.path.join(self.error_cache_dir, basename + ".pt")
            
            if os.path.exists(gt_path) and os.path.exists(err_path):
                self.samples.append((
                    os.path.join(self.tp_dir, f),
                    gt_path,
                    err_path
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path, err_path = self.samples[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        # Load GT mask → resize to 16x16
        gt = Image.open(gt_path).convert('L')
        gt_resized = gt.resize((16, 16), Image.NEAREST)
        gt_tensor = torch.from_numpy(np.array(gt_resized) > 128).float().unsqueeze(-1)  # [16, 16, 1]
        
        # Load precomputed prediction error map [256, 1280]
        error_map = torch.load(err_path)  # [256, 1280]
        
        return img, error_map, gt_tensor.view(-1, 1)  # [256, 1]


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--casia_dir', default='/home/uslib/quynhhuong/datasets/CASIA2/CASIA2')
    parser.add_argument('--checkpoint', default='/home/uslib/quynhhuong/ijepa/pretrained_models/IN1K-vit.h.14-300e.pth.tar')
    parser.add_argument('--output_dir', default='/home/uslib/quynhhuong/ijepa/forensic_outputs')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ============================================================
    # 1. Load dataset splits
    # ============================================================
    split_path = os.path.join(args.output_dir, 'dataset_splits.json')
    with open(split_path, 'r') as f:
        splits = json.load(f)
    print(f"Splits: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    
    # ============================================================
    # 2. Initialize Encoder (FROZEN)
    # ============================================================
    print("\nInitializing MultiScale Encoder...")
    from src.helper import init_model
    base_encoder, _ = init_model(
        device=device,
        model_name='vit_huge',
        patch_size=14,
        crop_size=224
    )
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    encoder_state = checkpoint['encoder']
    if all(k.startswith('module.') for k in encoder_state.keys()):
        encoder_state = {k.replace('module.', ''): v for k, v in encoder_state.items()}
    base_encoder.load_state_dict(encoder_state)
    
    encoder = MultiScaleEncoder(base_encoder, extract_layers=[4, 8, 12, 31])
    encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False  # Freeze encoder
    
    # ============================================================
    # 3. Initialize ForensicHead v4 (Gated Fusion with Latent Error)
    # ============================================================
    head = ForensicHead(error_dim=1280, use_predictor_error=True).to(device)
    print(f"\nForensicHead v4 — Latent Gated Fusion Architecture:")
    print(f"  Feature Branch: 5120 → 512 (multi-scale encoder features)")
    print(f"  Error Branch:   1280 → 1024 → 512 (latent error vector from World Model)")
    print(f"  Gate:           1024 → 512 (sigmoid, learnable contribution control)")
    print(f"  Classifier:     512 → 256 → 1")
    print(f"  Trainable params: {sum(p.numel() for p in head.parameters() if p.requires_grad):,}")
    
    # ============================================================
    # 4. Data Loaders
    # ============================================================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_cache = os.path.join(args.output_dir, 'error_cache_v4_latent', 'train')
    val_cache = os.path.join(args.output_dir, 'error_cache_v4_latent', 'val')
    
    train_ds = CASIAWorldModelDataset(args.casia_dir, splits['train'], train_cache, transform=transform)
    val_ds = CASIAWorldModelDataset(args.casia_dir, splits['val'], val_cache, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"\nTrain samples (with error cache): {len(train_ds)}")
    print(f"Val samples (with error cache):   {len(val_ds)}")
    
    if len(train_ds) == 0:
        print("ERROR: No training samples found. Did you run precompute_errors.py first?")
        return
    
    # ============================================================
    # 5. Optimization
    # ============================================================
    optimizer = optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # ============================================================
    # 6. Training Loop with Contribution Logging
    # ============================================================
    best_model_path = os.path.join(args.output_dir, 'forensic_head_v3_best.pt')
    print(f"\n{'='*70}")
    print(f"Starting World Model Forensic Training (Gated Fusion)")
    print(f"{'='*70}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch size: {args.batch_size}")
    print(f"Best model will be saved to: {best_model_path}")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # --- Train ---
        head.train()
        train_loss = 0
        epoch_gate_means = []
        epoch_gate_stds = []
        epoch_contrib_pcts = []
        
        for imgs, error_maps, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            imgs = imgs.to(device)
            error_maps = error_maps.to(device)    # [B, 256, 1]
            masks = masks.to(device)               # [B, 256, 1]
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                multi_feats = encoder(imgs)
            
            logits = head(multi_feats, error_map=error_maps)
            loss = criterion(logits, masks)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Collect contribution stats
            stats = head.get_contribution_stats()
            if stats:
                epoch_gate_means.append(stats['gate_mean'])
                epoch_gate_stds.append(stats['gate_std'])
                epoch_contrib_pcts.append(stats['contribution_pct'])
            
        # --- Validation ---
        head.eval()
        val_loss = 0
        val_gate_means = []
        val_contrib_pcts = []
        
        with torch.no_grad():
            for imgs, error_maps, masks in val_loader:
                imgs = imgs.to(device)
                error_maps = error_maps.to(device)
                masks = masks.to(device)
                multi_feats = encoder(imgs)
                logits = head(multi_feats, error_map=error_maps)
                val_loss += criterion(logits, masks).item()
                
                stats = head.get_contribution_stats()
                if stats:
                    val_gate_means.append(stats['gate_mean'])
                    val_contrib_pcts.append(stats['contribution_pct'])
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # --- Contribution Stats ---
        avg_gate_mean = np.mean(epoch_gate_means) if epoch_gate_means else 0
        avg_gate_std = np.mean(epoch_gate_stds) if epoch_gate_stds else 0
        avg_contrib = np.mean(epoch_contrib_pcts) if epoch_contrib_pcts else 0
        val_gate_mean = np.mean(val_gate_means) if val_gate_means else 0
        val_contrib = np.mean(val_contrib_pcts) if val_contrib_pcts else 0
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  --- Prediction Error Contribution (World Model) ---")
        print(f"  Train: Gate={avg_gate_mean:.4f}±{avg_gate_std:.4f} | Error Contribution={avg_contrib:.2f}%")
        print(f"  Val:   Gate={val_gate_mean:.4f}                   | Error Contribution={val_contrib:.2f}%")
        print(f"{'='*70}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(head.state_dict(), best_model_path)
            print(f"  ✓ Saved best model (Val Loss={avg_val_loss:.4f})")

    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Best model: {best_model_path}")
    print(f"  Final Prediction Error Contribution: {avg_contrib:.2f}% (train), {val_contrib:.2f}% (val)")
    print(f"{'='*70}")


if __name__ == '__main__':
    train()
