import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse

# Add path to I-JEPA source
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from src.models.multiscale_encoder import MultiScaleEncoder
from src.models.forensic_head import ForensicHead
from src.helper import init_model
from src.masks.utils import apply_masks

class ForensicDetectorV4:
    """
    Inference class for V4 Gated Fusion (1280-dim Latent Error).
    """
    def __init__(self, checkpoint_path, forensic_head_path, device='cuda:0'):
        self.device = device
        self.patch_size = 14
        self.img_size = 224
        self.grid_size = 16
        
        # 1. Initialize I-JEPA Models
        print("Initializing I-JEPA Encoder & Predictor...")
        self.base_encoder, self.predictor = init_model(
            device=device,
            model_name='vit_huge',
            patch_size=self.patch_size,
            crop_size=self.img_size,
            pred_depth=12
        )
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load encoder
        encoder_state = checkpoint['encoder']
        if all(k.startswith('module.') for k in encoder_state.keys()):
            encoder_state = {k.replace('module.', ''): v for k, v in encoder_state.items()}
        self.base_encoder.load_state_dict(encoder_state)
        self.base_encoder.eval()
        
        # Load predictor
        predictor_state = checkpoint['predictor']
        if all(k.startswith('module.') for k in predictor_state.keys()):
            predictor_state = {k.replace('module.', ''): v for k, v in predictor_state.items()}
        self.predictor.load_state_dict(predictor_state)
        self.predictor.eval()
        
        # MultiScale Wrapper
        self.encoder = MultiScaleEncoder(self.base_encoder, extract_layers=[4, 8, 12, 31])
        self.encoder.to(device)
        self.encoder.eval()
        
        # 2. Initialize Forensic Head (V4 - Gated Fusion)
        print("Loading ForensicHead V4...")
        # Note: hidden_dim=512, error_dim=1280 are defaults in current ForensicHead
        self.head = ForensicHead(error_dim=1280, use_predictor_error=True).to(device)
        head_state = torch.load(forensic_head_path, map_location=device)
        self.head.load_state_dict(head_state)
        self.head.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def detect(self, image_path):
        print(f"Processing: {image_path}")
        img = Image.open(image_path).convert('RGB')
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # A. Extract Multi-scale Features
            multi_feats = self.encoder(img_t)
            
            # B. Compute Latent Error Map (Leave-one-out)
            # This is the slow part (256 forward passes of predictor)
            print("Calculating 1280-dim latent error map (256 patches)...")
            full_rep = self.base_encoder(img_t)
            h_norm = F.layer_norm(full_rep, (full_rep.size(-1),))
            
            num_patches = self.grid_size * self.grid_size
            all_indices = list(range(num_patches))
            error_map = torch.zeros((1, num_patches, 1280), device=self.device)
            
            for idx in range(num_patches):
                ctx_indices = [i for i in all_indices if i != idx]
                ctx_mask = [torch.tensor(ctx_indices).unsqueeze(0).to(self.device)]
                tgt_mask = [torch.tensor([idx]).unsqueeze(0).to(self.device)]
                
                context_rep = apply_masks(full_rep, ctx_mask)
                target_pred = self.predictor(context_rep, ctx_mask, tgt_mask)
                target_gt = apply_masks(h_norm, tgt_mask)
                
                # V4 Error: Squared difference [1280]
                error_vec = (target_pred - target_gt).pow(2)
                error_map[0, idx] = error_vec.squeeze(0).squeeze(0)
            
            # C. Run Forensic Head
            logits = self.head(multi_feats, error_map=error_map) # [1, 256, 1]
            probs = torch.sigmoid(logits).view(16, 16)
            
            # Get contribution stats
            stats = self.head.get_contribution_stats()
            
        return probs.cpu().numpy(), stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='/home/uslib/quynhhuong/ijepa/forensic_outputs/test/Tp_D_CRN_M_N_ani10118_sec00098_11619.jpg')
    parser.add_argument('--checkpoint', default='/home/uslib/quynhhuong/ijepa/pretrained_models/IN1K-vit.h.14-300e.pth.tar')
    parser.add_argument('--head', default='/home/uslib/quynhhuong/ijepa/forensic_outputs/v4/checkpoints/forensic_head_v4_best.pt')
    parser.add_argument('--output', default='inference_results/test_v4_single.png')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    detector = ForensicDetectorV4(args.checkpoint, args.head, device=f'cuda:{args.gpu}')
    heatmap, stats = detector.detect(args.image_path)
    
    # --- Visualization ---
    plt.figure(figsize=(12, 5))
    
    # 1. Original
    plt.subplot(1, 2, 1)
    orig_img = Image.open(args.image_path).convert('RGB')
    plt.imshow(orig_img)
    plt.title("Original Image")
    plt.axis('off')
    
    # 2. Prediction Heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    plt.colorbar(label='Forgery Probability')
    
    title = f"V4 Latent Error Detection\nError Contribution: {stats['contribution_pct']:.2f}%"
    plt.title(title)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()
    
    print(f"\nDetection complete!")
    print(f"Prediction Error Contribution: {stats['contribution_pct']:.2f}%")
    print(f"Result saved to {args.output}")
