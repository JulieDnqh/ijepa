import torch
import torch.nn as nn
import torch.nn.functional as F

# nhận multi-scale features làm input
class ForensicHead(nn.Module):
    """
    Gated Fusion ForensicHead v3 - World Model Forensic Pipeline.
    
    Two-branch architecture that gives prediction error equal representation power:
      - Feature Branch: Multi-scale encoder features (5120-dim → 512-dim)
      - Error Branch: Prediction error map (1-dim → 512-dim)  
      - Gated Fusion: Learnable gate controls how much prediction error influences decision
    
    The gate value (0-1) directly indicates the contribution of prediction error
    to the final forgery detection, making the World Model's role explicit and measurable.
    """
    def __init__(self, embed_dim=1280, num_scales=4, hidden_dim=512,
                 error_dim=1280, use_predictor_error=True):
        super().__init__()
        self.use_predictor_error = use_predictor_error
        self.hidden_dim = hidden_dim
        self.error_dim = error_dim
        feature_dim = embed_dim * num_scales  # 5120
        
        # ============================================================
        # Feature Branch: Multi-scale encoder features → hidden_dim
        # ============================================================
        self.feature_branch = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # ============================================================
        # Error Branch: Prediction error (Latent Vector) → hidden_dim
        # Now handles 1280-dim vectors instead of 1-dim scalars.
        # ============================================================
        if use_predictor_error:
            self.error_branch = nn.Sequential(
                nn.Linear(error_dim, hidden_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            )
            
            # ============================================================
            # Learnable Gate: Controls contribution of prediction error
            # Input: concat(feature_out, error_out) → sigmoid → [0, 1]
            # ============================================================
            # sigmoid tạo trọng số điều chỉnh
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
        
        # ============================================================
        # Classification Head: Fused features → forgery probability
        # ============================================================
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Storage for contribution stats (updated during forward pass)
        self._gate_values = None
        self._feature_norm = None
        self._error_contribution_norm = None
        
    def forward(self, multiscale_features, error_map=None):
        """
        Args:
            multiscale_features: dict mapping scale_name -> [B, N, D]
            error_map: [B, N, error_dim] prediction error map (optional)
        Returns:
            logits: [B, N, 1]
        """
        # Collect and concatenate multi-scale features
        feats = []
        for name in ['layer_4', 'layer_8', 'layer_12', 'final']:
            feats.append(multiscale_features[name])
        # 4 layers × 1280-dim = 5120
        x = torch.cat(feats, dim=-1)  # [B, N, 5120]
        
        B, N, C = x.shape
        
        # Feature branch
        feat_out = self.feature_branch(x.view(B * N, C))  # [B*N, 512]
        
        if self.use_predictor_error and error_map is not None:
            # Error branch
            err_flat = error_map.view(B * N, self.error_dim)  # [B*N, 1280]
            err_out = self.error_branch(err_flat)  # [B*N, 512]
            
            # Gated fusion
            # bước 1: gate "quan sát" cả hai nhánh
            # gate nhìn thấy cả feature và error từ patch đó trước khi quyết định
            gate_input = torch.cat([feat_out, err_out], dim=-1)  # [B*N, 1024]
            # bước 2: sigmoid tạo trọng số điều chỉnh
            gate_values = self.gate(gate_input)  # [B*N, 512], values in [0, 1]
            
            # Fusion: feature + gate * error
            fused = feat_out + gate_values * err_out
            
            # Store stats for logging (detach to avoid affecting gradients)
            self._gate_values = gate_values.detach()
            self._feature_norm = feat_out.detach().norm(dim=-1).mean()
            self._error_contribution_norm = (gate_values * err_out).detach().norm(dim=-1).mean()
        else:
            fused = feat_out
            self._gate_values = None
            self._feature_norm = None
            self._error_contribution_norm = None
        
        # Classification
        logits = self.classifier(fused)  # [B*N, 1]
        
        return logits.view(B, N, 1)
    
    def get_contribution_stats(self):
        """
        Returns statistics about prediction error's contribution.
        Call after forward() to get stats for the last batch.
        
        Returns:
            dict with gate_mean, gate_std, feature_norm, error_contribution_norm, 
            contribution_ratio (% of fused signal coming from prediction error)
        """
        if self._gate_values is None:
            return None
        
        gate_mean = self._gate_values.mean().item()
        gate_std = self._gate_values.std().item()
        feat_norm = self._feature_norm.item()
        err_contrib_norm = self._error_contribution_norm.item()
        
        # Contribution ratio: how much of the fused signal comes from error
        total_norm = feat_norm + err_contrib_norm
        contribution_pct = (err_contrib_norm / total_norm * 100) if total_norm > 0 else 0
        
        return {
            'gate_mean': gate_mean,
            'gate_std': gate_std,
            'feature_norm': feat_norm,
            'error_contribution_norm': err_contrib_norm,
            'contribution_pct': contribution_pct
        }
