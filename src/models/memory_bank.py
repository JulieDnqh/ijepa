import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LatentMemoryBank:
    """
    Store patch-level features to identify 'known' world patterns.
    Used for Strategy 3 (Memory Bank) anomaly detection.
    """
    def __init__(self, feature_dim=1280, device='cuda'):
        self.feature_dim = feature_dim
        self.device = device
        self.memory = None # [M, D] tensor
        
    def build_from_features(self, all_features, coreset_ratio=0.01):
        """
        Build memory bank from a large set of patch features.
        Uses coreset sampling (random for now) to keep size manageable.
        
        Args:
            all_features: [N, D] tensor of patch features
            coreset_ratio: fraction of features to keep
        """
        N = all_features.shape[0]
        n_keep = max(1, int(N * coreset_ratio))
        
        print(f"Building memory bank: Sampling {n_keep} from {N} features...")
        
        # Simple random sampling for speed
        # PatchCore uses greedy coreset, but random is a good baseline
        indices = torch.randperm(N)[:n_keep]
        self.memory = all_features[indices].to(self.device)
        
        # Normalize for cosine similarity speedup
        self.memory = F.normalize(self.memory, dim=-1)
        print(f"Memory bank built. Shape: {self.memory.shape}")
        
    def query(self, patch_features, k=1):
        """
        Find nearest neighbor distance for each query patch.
        
        Args:
            patch_features: [B, N, D] tensor
            k: number of neighbors to consider (default 1 for anomaly score)
        Returns:
            anomaly_score: [B, N] tensor (1 - max_similarity)
        """
        if self.memory is None:
            raise ValueError("Memory bank not built. Call build_from_features or load first.")
            
        B, N, D = patch_features.shape
        # Flatten batch and patches
        queries = patch_features.reshape(-1, D)
        queries = F.normalize(queries, dim=-1)
        
        # Compute cosine similarity
        # [BN, D] @ [D, M] -> [BN, M]
        sims = torch.mm(queries, self.memory.t())
        
        # Find max similarity per query
        max_sims, _ = sims.topk(k, dim=-1)
        
        # Anomaly score is 1 - similarity (range 0 to 2, usually 0 to 1 for aligned features)
        scores = 1.0 - max_sims[:, 0]
        
        return scores.reshape(B, N)
        
    def save(self, path):
        torch.save(self.memory, path)
        print(f"Memory bank saved to {path}")
        
    def load(self, path):
        self.memory = torch.load(path, map_location=self.device)
        print(f"Memory bank loaded from {path}. Shape: {self.memory.shape}")
