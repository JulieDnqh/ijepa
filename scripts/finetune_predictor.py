"""
Fine-tune I-JEPA (Encoder + Predictor) on authentic CASIA images.

This adapts the original I-JEPA self-supervised training loop for single-GPU
fine-tuning on authentic images only, so the Predictor learns the distribution
of "real" CASIA images. This makes prediction errors on tampered regions 
more pronounced.

Key differences from original I-JEPA training:
  - Single GPU (no DDP)
  - Fine-tuning from pretrained checkpoint (not from scratch) 
  - Only authentic images (Au/) from the CASIA train split
  - Fewer epochs (domain adaptation, not full training)
"""
import os
import copy
import json
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import argparse
import logging

from src.helper import init_model
from src.masks.multiblock import MaskCollator
from src.masks.utils import apply_masks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CASIAAuthenticDataset(Dataset):
    """Dataset of only authentic (real) CASIA images for self-supervised training."""
    
    def __init__(self, casia_dir, file_list=None, transform=None):
        self.transform = transform
        au_dir = os.path.join(casia_dir, 'Au')
        
        if file_list is not None:
            # Use provided file list (from train split)
            self.images = [os.path.join(au_dir, f) for f in file_list 
                          if os.path.exists(os.path.join(au_dir, f))]
        else:
            # Use all authentic images
            self.images = [os.path.join(au_dir, f) for f in sorted(os.listdir(au_dir))
                          if f.lower().endswith(('.jpg', '.png', '.bmp', '.tif'))]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return (img,)  # Return as tuple to match MaskCollator expectation


def finetune_predictor(args):
    device = f'cuda:{args.gpu}'
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ========================================
    # 1. Get authentic images from train split
    # ========================================
    au_dir = os.path.join(args.casia_dir, 'Au')
    all_au_files = sorted([f for f in os.listdir(au_dir)
                          if f.lower().endswith(('.jpg', '.png', '.bmp', '.tif'))])
    
    # Use the same split seed for consistency: take 80% for training
    rng = np.random.RandomState(42) 
    rng.shuffle(all_au_files)
    n_train = int(len(all_au_files) * 0.8)
    train_au_files = all_au_files[:n_train]
    val_au_files = all_au_files[n_train:n_train + int(len(all_au_files) * 0.1)]
    
    print(f"Authentic images: Total={len(all_au_files)}, Train={len(train_au_files)}, Val={len(val_au_files)}")
    
    # ========================================
    # 2. Initialize Models
    # ========================================
    print("\nInitializing I-JEPA models...")
    encoder, predictor = init_model(
        device=device,
        model_name='vit_huge',
        patch_size=14,
        crop_size=224,
        pred_depth=12
    )
    
    # Load pretrained checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    encoder_state = checkpoint['encoder']
    if all(k.startswith('module.') for k in encoder_state.keys()):
        encoder_state = {k.replace('module.', ''): v for k, v in encoder_state.items()}
    encoder.load_state_dict(encoder_state)
    
    predictor_state = checkpoint['predictor']
    if all(k.startswith('module.') for k in predictor_state.keys()):
        predictor_state = {k.replace('module.', ''): v for k, v in predictor_state.items()}
    predictor.load_state_dict(predictor_state)
    
    # Create target encoder (EMA copy of encoder)
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False
    
    # Freeze encoder — only fine-tune predictor
    for p in encoder.parameters():
        p.requires_grad = False
    
    encoder.eval()
    target_encoder.eval()
    predictor.train()
    
    trainable_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    print(f"Predictor trainable params: {trainable_params:,}")
    
    # ========================================
    # 3. Data
    # ========================================
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Mask collator — same as original I-JEPA
    mask_collator = MaskCollator(
        input_size=(224, 224),
        patch_size=14,
        pred_mask_scale=(0.15, 0.2),  # target blocks: 15-20% of patches
        enc_mask_scale=(0.85, 1.0),   # context blocks: 85-100% of patches
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        allow_overlap=False,
        min_keep=10
    )
    
    val_mask_collator = MaskCollator(
        input_size=(224, 224),
        patch_size=14,
        pred_mask_scale=(0.15, 0.2),
        enc_mask_scale=(0.85, 1.0),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        allow_overlap=False,
        min_keep=10
    )
    
    train_ds = CASIAAuthenticDataset(args.casia_dir, train_au_files, transform)
    val_ds = CASIAAuthenticDataset(args.casia_dir, val_au_files, val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, collate_fn=mask_collator, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=val_mask_collator, drop_last=True)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # ========================================
    # 4. Optimizer (only predictor params)
    # ========================================
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=args.lr,
        weight_decay=0.05
    )
    
    # EMA momentum for target encoder
    ema_momentum = 0.996
    
    # ========================================
    # 5. Training Loop
    # ========================================
    best_val_loss = float('inf')
    best_path = os.path.join(args.output_dir, 'ijepa_casia_finetuned_best.pt')
    
    print(f"\nStarting Predictor fine-tuning on authentic CASIA images...")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch size: {args.batch_size}")
    print(f"Best model will be saved to: {best_path}")
    
    for epoch in range(args.epochs):
        # --- Train ---
        predictor.train()
        train_loss = 0
        n_batches = 0
        
        for batch_data, masks_enc, masks_pred in tqdm(train_loader, 
                                                        desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            imgs = batch_data[0].to(device)
            masks_enc = [m.to(device) for m in masks_enc]
            masks_pred = [m.to(device) for m in masks_pred]
            
            # Forward target (no grad)
            with torch.no_grad():
                h = target_encoder(imgs)
                h = F.layer_norm(h, (h.size(-1),))
                B = len(h)
                h = apply_masks(h, masks_pred)
                # repeat for multiple encoder masks
                h = h.repeat(len(masks_enc), 1, 1) if len(masks_enc) > 1 else h
            
            # Forward context (predictor is trainable)
            with torch.no_grad():
                z = encoder(imgs, masks_enc)
            z = predictor(z, masks_enc, masks_pred)
            
            # Loss: smooth L1 (same as original I-JEPA)
            loss = F.smooth_l1_loss(z, h)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # EMA update of target encoder from encoder 
            # (encoder is frozen, so target_encoder stays close to it)
            with torch.no_grad():
                for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                    param_k.data.mul_(ema_momentum).add_((1. - ema_momentum) * param_q.data)
            
            train_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = train_loss / max(n_batches, 1)
        
        # --- Validation ---
        predictor.eval()
        val_loss = 0
        n_val = 0
        
        with torch.no_grad():
            for batch_data, masks_enc, masks_pred in val_loader:
                imgs = batch_data[0].to(device)
                masks_enc = [m.to(device) for m in masks_enc]
                masks_pred = [m.to(device) for m in masks_pred]
                
                h = target_encoder(imgs)
                h = F.layer_norm(h, (h.size(-1),))
                h = apply_masks(h, masks_pred)
                h = h.repeat(len(masks_enc), 1, 1) if len(masks_enc) > 1 else h
                
                z = encoder(imgs, masks_enc)
                z = predictor(z, masks_enc, masks_pred)
                
                val_loss += F.smooth_l1_loss(z, h).item()
                n_val += 1
        
        avg_val_loss = val_loss / max(n_val, 1)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save full checkpoint (encoder + predictor + target_encoder)
            torch.save({
                'encoder': encoder.state_dict(),
                'predictor': predictor.state_dict(),
                'target_encoder': target_encoder.state_dict(),
                'epoch': epoch + 1,
                'val_loss': avg_val_loss,
            }, best_path)
            print(f"  ✓ Saved best model (Val Loss={avg_val_loss:.6f})")
    
    print(f"\nFine-tuning complete. Best Val Loss: {best_val_loss:.6f}")
    print(f"Best model: {best_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--casia_dir', default='/home/uslib/quynhhuong/datasets/CASIA2/CASIA2')
    parser.add_argument('--checkpoint', default='/home/uslib/quynhhuong/ijepa/pretrained_models/IN1K-vit.h.14-300e.pth.tar')
    parser.add_argument('--output_dir', default='/home/uslib/quynhhuong/ijepa/forensic_outputs')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    finetune_predictor(args)
