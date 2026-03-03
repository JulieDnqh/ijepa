import torch
import torch.nn as nn

class MultiScaleEncoder(nn.Module):
    """
    Wrap I-JEPA ViT Encoder to extract intermediate features from specific layers.
    Used for Strategy 2 (Hierarchical Multi-scale) detection.
    """
    def __init__(self, base_encoder, extract_layers=[4, 8, 12, 31]):
        super().__init__()
        self.encoder = base_encoder
        self.extract_layers = extract_layers
        self.features = {}
        self.hooks = []
        
        # Register forward hooks on specified transformer blocks
        for i, block in enumerate(self.encoder.blocks):
            if i in extract_layers:
                # Use a closure to capture the layer index correctly
                def get_hook(idx):
                    def hook(module, input, output):
                        self.features[idx] = output
                    return hook
                
                handle = block.register_forward_hook(get_hook(i))
                self.hooks.append(handle)
                
    def forward(self, x, masks=None):
        self.features = {}
        # Forward through the entire encoder
        final_output = self.encoder(x, masks=masks)
        
        # Return dict mapping layer index to features [B, N, D]
        # Layer indices: 4, 8, 12, 31 (semantic)
        # 'final' corresponds to the normalized output after all blocks
        return {
            'final': final_output,
            **{f'layer_{k}': v for k, v in self.features.items()}
        }

    def remove_hooks(self):
        """Clean up hooks if needed."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
