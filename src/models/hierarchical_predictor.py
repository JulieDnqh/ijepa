import torch
import torch.nn as nn
from src.models.vision_transformer import VisionTransformerPredictor
from src.masks.utils import apply_masks

class HierarchicalPredictor(nn.Module):
    """
    A collection of Predictors, each specialized for a different hierarchical level
    (layer) of the ViT encoder.
    """
    def __init__(self, num_patches=256, embed_dim=1280, pred_dim=384, 
                 scales=['layer_4', 'layer_8', 'layer_12', 'final'],
                 depths=4):
        super().__init__()
        self.predictors = nn.ModuleDict()
        self.scales = scales
        
        # depths can be an int (default for all) or a dict mapping scale to depth
        for scale in scales:
            d = depths[scale] if isinstance(depths, dict) else depths
            
            self.predictors[scale] = VisionTransformerPredictor(
                num_patches=num_patches,
                embed_dim=embed_dim,
                predictor_embed_dim=pred_dim,
                depth=d,
                num_heads=6
            )
            
    def forward(self, multiscale_features, context_masks, target_masks, scale_name=None):
        """
        Args:
            multiscale_features: dict scale -> [B, N, D]
            context_masks: list of context indices
            target_masks: list of target indices to predict
            scale_name: if provided, only run predictor for this specific scale
        """
        if scale_name is not None:
            if scale_name not in self.predictors:
                raise ValueError(f"Scale {scale_name} not found in predictors.")
            
            feat = multiscale_features[scale_name]
            # Must apply masks to get context tokens before passing to predictor
            ctx_feat = apply_masks(feat, context_masks)
            return self.predictors[scale_name](ctx_feat, context_masks, target_masks)
        
        # Run for all scales
        results = {}
        for name, predictor in self.predictors.items():
            feat = multiscale_features[name]
            ctx_feat = apply_masks(feat, context_masks)
            results[name] = predictor(ctx_feat, context_masks, target_masks)
            
        return results
