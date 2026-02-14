import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
import sys
import argparse

# Add path to I-JEPA source (allowing imports from src.xxx)
# Since the script is in ijepa/scripts, the root is one level up
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

print("Starting I-JEPA Inference Test Script...")
print(f"Python: {sys.executable}")
print(f"CWD: {os.getcwd()}")
print(f"I-JEPA Root: {ROOT_DIR}")

from src.models.vision_transformer import vit_huge, vit_predictor
from src.helper import init_model
from src.masks.utils import apply_masks
from src.masks.multiblock import MaskCollator
print("Imports from src successful")

class IJEPAInferenceTest:
    def __init__(self, device='cuda', checkpoint_path=None):
        self.device = device
        self.patch_size = 14
        self.img_size = 224
        self.grid_size = self.img_size // self.patch_size # 16
        
        # 1. Initialize models (ViT-H/14)
        self.encoder, self.predictor = init_model(
            device=device,
            patch_size=self.patch_size,
            model_name='vit_huge',
            crop_size=self.img_size,
            pred_depth=12,
            pred_emb_dim=384
        )
        
        # 2. Load Checkpoint
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Encoder
            encoder_state = checkpoint['encoder']
            if any(k.startswith('module.') for k in encoder_state.keys()):
                encoder_state = {k.replace('module.', ''): v for k, v in encoder_state.items()}
            self.encoder.load_state_dict(encoder_state)
            
            # Predictor
            predictor_state = checkpoint['predictor']
            if any(k.startswith('module.') for k in predictor_state.keys()):
                predictor_state = {k.replace('module.', ''): v for k, v in predictor_state.items()}
            self.predictor.load_state_dict(predictor_state)
            
            print(f"Loaded checkpoint from {checkpoint_path}")
            
        self.encoder.eval()
        self.predictor.eval()
        
        # 3. Setup Transform with Padding
        self.transform = self.get_padding_transform()

    def get_padding_transform(self):
        class PadToSquare(object):
            def __init__(self, size, fill=0):
                self.size = size
                self.fill = fill
            def __call__(self, img):
                # padding
                w, h = img.size
                # lấy cạnh dài nhất làm chuẩn
                scale = self.size / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                
                # Create black background
                new_img = Image.new("RGB", (self.size, self.size), (self.fill, self.fill, self.fill))
                # Paste centered
                # nếu h=224 thì top=0
                top = (self.size - new_h) // 2
                # nếu w=224 thì left=0
                left = (self.size - new_w) // 2
                new_img.paste(img, (left, top))
                return new_img

        return transforms.Compose([
            PadToSquare(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_mask_indices(self, coords_list):
        """
        coords_list: list of (top, left, height, width) tuples
        """
        # full mask to identify context
        full_mask = torch.zeros((self.grid_size, self.grid_size), dtype=torch.int32)
        target_idx_list = []
        
        for top, left, height, width in coords_list:
            block_mask = torch.zeros((self.grid_size, self.grid_size), dtype=torch.int32)
            # Ensure indices stay within grid
            h_end = min(top + height, self.grid_size)
            w_end = min(left + width, self.grid_size)
            block_mask[top:h_end, left:w_end] = 1
            
            idx = torch.nonzero(block_mask.flatten()).squeeze()
            if idx.dim() == 0: idx = idx.unsqueeze(0)
            target_idx_list.append(idx.to(self.device))
            full_mask[top:h_end, left:w_end] = 1
            
        target_all_idx = torch.nonzero(full_mask.flatten()).squeeze().to(self.device).unsqueeze(0)
        
        context_mask = 1 - full_mask
        context_indices = torch.nonzero(context_mask.flatten()).squeeze().to(self.device).unsqueeze(0)
        
        return context_indices, target_idx_list, target_all_idx

    def get_object_mask_indices(self, image_path, target_class='zebra', ratio=1.0):
        from ultralytics import YOLO
        import random
        
        yolo_model = YOLO("/home/uslib/quynhhuong/ijepa/yolov8n-seg.pt")
        
        orig_img = Image.open(image_path).convert('RGB')
        padded_img = self.transform.transforms[0](orig_img) # PadToSquare(224)
        
        results = yolo_model(padded_img)
        
        object_mask = None
        for result in results:
            if result.masks is not None and result.boxes is not None:
                for i, cls_idx in enumerate(result.boxes.cls):
                    name = result.names[int(cls_idx)]
                    if name == target_class:
                        object_mask = result.masks.data[i] 
                        break
        
        if object_mask is None:
            print(f"Warning: No {target_class} found.")
            return None, None

        object_mask = object_mask.float()
        object_mask = F.interpolate(object_mask.unsqueeze(0).unsqueeze(0), size=(self.img_size, self.img_size), mode='nearest').squeeze()
        
        grid_mask = torch.zeros((self.grid_size, self.grid_size), device=self.device)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                patch = object_mask[r*self.patch_size:(r+1)*self.patch_size, c*self.patch_size:(c+1)*self.patch_size]
                if patch.mean() > 0.4:
                    grid_mask[r, c] = 1
        
        object_indices = torch.nonzero(grid_mask.flatten()).squeeze().tolist()
        if isinstance(object_indices, int): object_indices = [object_indices]
        
        if not object_indices:
            return None, None
            
        # Commented out ratio-based selection as per user request
        # random.seed(42)
        # num_to_mask = int(len(object_indices) * ratio)
        # masked_indices = random.sample(object_indices, num_to_mask)
        masked_indices = object_indices # Use all detected object patches
        
        all_indices = set(range(self.grid_size * self.grid_size))
        context_indices = list(all_indices - set(masked_indices))
        
        return torch.tensor(context_indices).unsqueeze(0).to(self.device), torch.tensor(masked_indices).unsqueeze(0).to(self.device)

    def get_multi_block_indices(self, npred=4, seed=None):
        collator = MaskCollator(
            input_size=(self.img_size, self.img_size),
            patch_size=self.patch_size,
            nenc=1,
            npred=npred,
            enc_mask_scale=(0.85, 1.0), # Context occupies most of the image
            pred_mask_scale=(0.15, 0.2), # Each target block is ~15-20%
            allow_overlap=False
        )
        
        if seed is not None:
             import multiprocessing
             # Reset counter to get deterministic seed if needed, though step() uses Value
             # For inference, we'll just use a dummy batch to trigger collator
        
        dummy_batch = [torch.zeros(3, self.img_size, self.img_size)]
        _, masks_enc, masks_pred = collator(dummy_batch)
        
        # masks_enc: [B, nenc, C_patches] -> list of list
        # masks_pred: [B, npred, T_patches] -> list of list
        
        # masks_pred: [B, npred, T_patches] -> list of list (if collated, tensor)
        
        # We take first image in batch (B=1)
        enc_indices = masks_enc[0][0].to(self.device) # First enc mask
        
        # target_idx_list is typically a tensor [npred, patches] after collation
        target_idx_list = masks_pred[0].to(self.device)
        
        # Combine all prediction masks for target_gt calculation
        all_pred_indices = target_idx_list.flatten().to(self.device)
        
        return enc_indices.unsqueeze(0), target_idx_list, all_pred_indices.unsqueeze(0)

    def run_test(self, image_path, output_path, mask_type='block', mask_coords=None, target_class='zebra', ratio=1.0, npred=4, seed=None):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        orig_img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(orig_img).unsqueeze(0).to(self.device)
        
        if mask_type == 'object':
            context_idx, target_idx = self.get_object_mask_indices(image_path, target_class, ratio)
            target_all_idx = target_idx
        elif mask_type == 'multiblock':
            context_idx, target_idx_list, target_all_idx = self.get_multi_block_indices(npred=npred, seed=seed)
            target_idx = target_idx_list # list of tensors
        else: # block mode
            if isinstance(mask_coords, str):
                # Parse multiple blocks: "7,7,4,7 ; 2,2,3,3"
                blocks = []
                for b_str in mask_coords.split(';'):
                    if b_str.strip():
                        blocks.append([int(p) for p in b_str.split(',')])
                coords_list = blocks
            elif isinstance(mask_coords, list) and isinstance(mask_coords[0], list):
                coords_list = mask_coords
            else:
                coords_list = [mask_coords] # single block list [top, left, h, w]
                
            context_idx, target_idx_list, target_all_idx = self.get_mask_indices(coords_list)
            target_idx = target_idx_list
        
        with torch.no_grad():
            full_rep = self.encoder(img_tensor)
            h_norm = F.layer_norm(full_rep, (full_rep.size(-1),))
            
            # Using target_all_idx (combined mask) to handle variable block sizes
            target_gt = apply_masks(h_norm, target_all_idx)
            context_rep = apply_masks(full_rep, context_idx)
            target_pred = self.predictor(context_rep, context_idx, target_all_idx)
            
            cos_sim = F.cosine_similarity(target_gt, target_pred, dim=-1)
            mean_sim = cos_sim.mean().item()
            std_sim = cos_sim.std().item()
            
            label = f"{mask_type}"
            if mask_type == 'object': label += f" {ratio*100}%"
            elif mask_type == 'multiblock': label += f" npred={npred}"
            
            print(f"[{label}] Mean Cosine Similarity: {mean_sim:.4f}")
            
        self.visualize(img_tensor, target_all_idx, output_path, mean_sim, std_sim, cos_sim.flatten(), mask_type, ratio if mask_type=='object' else npred)

    def visualize(self, img_tensor, target_idx, output_path, mean_sim, std_sim, cos_sim_flat, mask_type, ratio):
        img = img_tensor[0].cpu().permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        masked_img = img.copy()
        for idx in target_idx[0]:
            r, c = idx // self.grid_size, idx % self.grid_size
            masked_img[r*self.patch_size:(r+1)*self.patch_size, c*self.patch_size:(c+1)*self.patch_size] = 0
            
        heatmap = np.ones((self.grid_size, self.grid_size)) * np.nan
        for idx, val in zip(target_idx[0], cos_sim_flat):
            r, c = idx // self.grid_size, idx % self.grid_size
            heatmap[r, c] = val.item()
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img); axes[0].set_title("Original Image"); axes[0].axis('off')
        axes[1].imshow(masked_img); axes[1].set_title(f"Context (Masked {mask_type})"); axes[1].axis('off')
        
        im = axes[2].imshow(heatmap, vmin=0, vmax=1.0, cmap='RdYlGn')
        axes[2].set_title(f"Similarity Heatmap\nMean: {mean_sim:.4f}")
        fig.colorbar(im, ax=axes[2], label='Cosine Similarity')
        
        axes[2].set_xticks(np.arange(self.grid_size))
        axes[2].set_yticks(np.arange(self.grid_size))
        axes[2].set_xticklabels(range(self.grid_size))
        axes[2].set_yticklabels(range(self.grid_size))
        axes[2].set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        axes[2].set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        axes[2].grid(which='minor', color='gray', linestyle='-', linewidth=1)
        axes[2].tick_params(which='minor', bottom=False, left=False)
        
        if mask_type == 'object':
            title_suffix = f"{ratio*100:.0f}%"
        elif mask_type == 'multiblock':
            title_suffix = f"npred={ratio}"
        else: # block
            title_suffix = f"manual"
            
        plt.suptitle(f"I-JEPA Predictor Inference: {mask_type} ({title_suffix})\nMean Sim: {mean_sim:.4f}", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    size_100_percent="7,7,4,7"
    size_75_percent="7,7,4,5"
    size_50_percent="7,7,4,3"
    size_25_percent="8,12,3,2"

    output_100_percent="/home/uslib/quynhhuong/ijepa/inference_results/test_01_100_percent.png"
    output_75_percent="/home/uslib/quynhhuong/ijepa/inference_results/test_01_75_percent.png"
    output_50_percent="/home/uslib/quynhhuong/ijepa/inference_results/test_01_50_percent.png"
    output_25_percent="/home/uslib/quynhhuong/ijepa/inference_results/test_01_25_percent.png"

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="/home/uslib/quynhhuong/ijepa/test/test_01.jpg")
    parser.add_argument("--checkpoint", type=str, default="/home/uslib/quynhhuong/ijepa/pretrained_models/IN1K-vit.h.14-300e.pth.tar")
    parser.add_argument("--mask_type", type=str, default="block", choices=['block', 'object', 'multiblock'])
    # parser.add_argument("--mask_coords", type=str, default="7,7,4,7", help="top,left,h,w")
    # parser.add_argument("--mask_coords", type=str, default=size_100_percent, help="top,left,h,w")
    # parser.add_argument("--mask_coords", type=str, default=size_75_percent, help="top,left,h,w")
    # parser.add_argument("--mask_coords", type=str, default=size_50_percent, help="top,left,h,w")
    parser.add_argument("--mask_coords", type=str, default=size_25_percent, help="top,left,h,w")
    parser.add_argument("--ratio", type=float, default=1.0, help="Ratio for object masking (0.0-1.0). (Commented out in logic)")
    parser.add_argument("--npred", type=int, default=4, help="Number of target blocks for multiblock masking")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--target_class", type=str, default="zebra")
    # parser.add_argument("--output", type=str, default="/home/uslib/quynhhuong/ijepa/inference_results/test_01_result.png")
    # parser.add_argument("--output", type=str, default=output_100_percent)
    # parser.add_argument("--output", type=str, default=output_75_percent)
    # parser.add_argument("--output", type=str, default=output_50_percent)
    parser.add_argument("--output", type=str, default=output_25_percent)
    args = parser.parse_args()
    
    test = IJEPAInferenceTest(checkpoint_path=args.checkpoint)
    
    # Commented out multi-ratio test loop for object masking
    # if args.mask_type == 'object' and args.ratio == 0:
    #     ratios = [0.25, 0.50, 0.75, 1.0]
    #     for r in ratios:
    #         out = args.output.replace(".png", f"_obj_{int(r*100)}.png")
    #         test.run_test(args.image, out, mask_type='object', target_class=args.target_class, ratio=r)
    # else:

    test.run_test(
        args.image, 
        args.output, 
        mask_type=args.mask_type, 
        mask_coords=args.mask_coords, # Pass raw string for multi-block parsing
        target_class=args.target_class, 
        ratio=args.ratio,
        npred=args.npred,
        seed=args.seed
    )
