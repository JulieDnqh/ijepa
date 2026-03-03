"""
Precompute I-JEPA Prediction Error Maps for Forensic Training.

Uses leave-one-out masking: for each patch, mask it out and predict it
from the remaining context. The prediction error (1 - cosine_similarity)
is the World Model's "surprise" signal.

Caches results as .pt files for fast loading during training.
"""
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import argparse

from src.helper import init_model
from src.models.multiscale_encoder import MultiScaleEncoder
from src.masks.utils import apply_masks


def precompute_errors(args):
    device = f'cuda:{args.gpu}'
    
    # --- Load split file ---
    with open(args.split_file, 'r') as f:
        splits = json.load(f)
    
    # Determine which split(s) to process
    if args.split == 'all':
        file_lists = {
            'train': splits['train'],
            'val': splits['val'],
        }
    else:
        file_lists = {args.split: splits[args.split]}
    
    # --- Initialize Models ---
    print("Initializing I-JEPA Encoder...")
    base_encoder, predictor = init_model(
        device=device,
        model_name='vit_huge',
        patch_size=14,
        crop_size=224,
        pred_depth=12
    )
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Load encoder
    encoder_state = checkpoint['encoder']
    if all(k.startswith('module.') for k in encoder_state.keys()):
        encoder_state = {k.replace('module.', ''): v for k, v in encoder_state.items()}
    base_encoder.load_state_dict(encoder_state)
    base_encoder.eval()
    
    # Load predictor
    predictor_state = checkpoint['predictor']
    if all(k.startswith('module.') for k in predictor_state.keys()):
        predictor_state = {k.replace('module.', ''): v for k, v in predictor_state.items()}
    predictor.load_state_dict(predictor_state)
    predictor.eval()
    
    # MultiScale Encoder (for feature extraction)
    encoder = MultiScaleEncoder(base_encoder, extract_layers=[4, 8, 12, 31])
    encoder.to(device)
    encoder.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    grid_size = 16
    num_patches = grid_size * grid_size  # 256
    all_indices = list(range(num_patches))
    tp_dir = os.path.join(args.casia_dir, 'Tp')
    
    # --- Process each split ---
    for split_name, file_list in file_lists.items():
        cache_dir = os.path.join(args.output_dir, 'error_cache_v4_latent', split_name)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check how many are already done
        already_done = set(os.listdir(cache_dir))
        remaining = [f for f in file_list 
                     if os.path.splitext(f)[0] + '.pt' not in already_done]
        
        # Apply sharding for multi-GPU parallel
        if args.num_shards > 1:
            remaining = [f for i, f in enumerate(remaining) if i % args.num_shards == args.shard]
        
        print(f"\n{'='*50}")
        print(f"Split: {split_name} | Total: {len(file_list)} | "
              f"Already cached: {len(file_list)-len(remaining)} | Remaining: {len(remaining)}")
        print(f"Cache dir: {cache_dir}")
        print(f"{'='*50}")
        
        for fi, fname in enumerate(tqdm(remaining, desc=f"Precomputing [{split_name}]")):
            img_path = os.path.join(tp_dir, fname)
            basename = os.path.splitext(fname)[0]
            save_path = os.path.join(cache_dir, basename + '.pt')
            
            try:
                img = Image.open(img_path).convert('RGB')
                img_t = transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    # Get full representation from encoder
                    full_rep = base_encoder(img_t)  # [1, 256, 1280]
                    h_norm = F.layer_norm(full_rep, (full_rep.size(-1),))
                
                # Leave-one-out prediction error vector (1280-dim per patch)
                # Output shape: [256, 1280]
                error_map = torch.zeros((num_patches, 1280), device='cpu')
                
                with torch.no_grad():
                    for idx in range(num_patches):
                        ctx_indices = [i for i in all_indices if i != idx]
                        ctx_mask = [torch.tensor(ctx_indices).unsqueeze(0).to(device)]
                        tgt_mask = [torch.tensor([idx]).unsqueeze(0).to(device)]
                        
                        # Context representation
                        context_rep = apply_masks(full_rep, ctx_mask)
                        
                        # Predictor predicts target from context
                        target_pred = predictor(context_rep, ctx_mask, tgt_mask) # [1, 1, 1280]
                        
                        # Ground truth target
                        target_gt = apply_masks(h_norm, tgt_mask) # [1, 1, 1280]
                        
                        # Prediction error vector = (pred - gt)^2 
                        # We use squared difference to capture component-wise surprise
                        error_vec = (target_pred - target_gt).pow(2).squeeze(0).squeeze(0) # [1280]
                        error_map[idx] = error_vec.cpu()
                
                # Save: error_map [256, 1280] as a tensor
                torch.save(error_map, save_path)
                
            except Exception as e:
                print(f"\nError processing {fname}: {e}")
                continue
        
        print(f"Done: {split_name} ({len(file_list)} images cached at {cache_dir})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--casia_dir', default='/home/uslib/quynhhuong/datasets/CASIA2/CASIA2')
    parser.add_argument('--checkpoint', default='/home/uslib/quynhhuong/ijepa/pretrained_models/IN1K-vit.h.14-300e.pth.tar')
    parser.add_argument('--split_file', default='/home/uslib/quynhhuong/ijepa/forensic_outputs/dataset_splits.json')
    parser.add_argument('--output_dir', default='/home/uslib/quynhhuong/ijepa/forensic_outputs')
    parser.add_argument('--split', default='all', choices=['train', 'val', 'all'],
                        help='Which split to precompute (default: train+val)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--shard', type=int, default=0,
                        help='Shard index (0-based) for multi-GPU parallel')
    parser.add_argument('--num_shards', type=int, default=1,
                        help='Total number of shards (= number of GPUs)')
    args = parser.parse_args()
    
    precompute_errors(args)
