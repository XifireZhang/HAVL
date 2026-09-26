import torch
import torch.nn as nn
import torch.nn.functional as F

class CapsuleNetwork(nn.Module):
    """
    Capsule Network for Multi-Interest Extraction
    
    Uses dynamic routing to extract multiple interest representations from history items.
    
    Architecture:
    - Input: history item embeddings (B, max_H, enc_dim)
    - Bilinear mapping: enc_dim → capsule_dim for each capsule
    - Dynamic routing: iteratively refine capsule weights
    - Output: multiple interest capsules (B, num_interests, capsule_dim)
    
    Args:
        enc_dim: input embedding dimension
        num_interests: number of interest capsules to extract (default: 3)
        capsule_dim: dimension of each capsule (default: enc_dim)
        num_routing: number of dynamic routing iterations (default: 3)
    """
    def __init__(self, enc_dim, num_interests=3, capsule_dim=None, num_routing=3):
        super(CapsuleNetwork, self).__init__()
        self.enc_dim = enc_dim
        self.num_interests = num_interests
        self.capsule_dim = capsule_dim if capsule_dim is not None else enc_dim
        self.num_routing = num_routing
        
        # Bilinear mapping: transform each history item to capsule space
        # For each interest capsule, we have a separate transformation matrix
        self.bilinear = nn.Parameter(
            torch.randn(num_interests, enc_dim, self.capsule_dim) * 0.01
        )
        
    def forward(self, history_emb, history_mask=None):
        """
        @input:
            history_emb: (B, max_H, enc_dim) - history item embeddings
            history_mask: (B, max_H) - binary mask for valid history items
        @output:
            interests: (B, num_interests, capsule_dim) - extracted interest capsules
        """
        B, max_H, enc_dim = history_emb.shape
        
        # Transform history items to capsule space
        # history_emb: (B, max_H, enc_dim)
        # bilinear: (num_interests, enc_dim, capsule_dim)
        # u_hat: (B, max_H, num_interests, capsule_dim)
        u_hat = torch.einsum('bhe,iec->bhic', history_emb, self.bilinear)
        
        # Initialize routing logits
        # b: (B, max_H, num_interests)
        b = torch.zeros(B, max_H, self.num_interests, device=history_emb.device)
        
        # Apply mask to routing logits if provided
        if history_mask is not None:
            # history_mask: (B, max_H) -> (B, max_H, 1)
            mask = history_mask.unsqueeze(-1).float()
            b = b.masked_fill(mask == 0, -1e9)
        
        # Dynamic routing
        for iteration in range(self.num_routing):
            # Softmax over interests dimension
            # c: (B, max_H, num_interests)
            c = F.softmax(b, dim=-1)
            
            # Weighted sum of transformed embeddings
            # c: (B, max_H, num_interests, 1)
            # u_hat: (B, max_H, num_interests, capsule_dim)
            # s: (B, num_interests, capsule_dim)
            s = torch.sum(c.unsqueeze(-1) * u_hat, dim=1)
            
            # Squash activation (non-linear normalization)
            # v: (B, num_interests, capsule_dim)
            v = self.squash(s)
            
            # Update routing logits (except for last iteration)
            if iteration < self.num_routing - 1:
                # Agreement: dot product between capsule output and transformed input
                # v: (B, num_interests, capsule_dim)
                # u_hat: (B, max_H, num_interests, capsule_dim)
                # delta_b: (B, max_H, num_interests)
                delta_b = torch.sum(v.unsqueeze(1) * u_hat, dim=-1)
                b = b + delta_b
                
                # Re-apply mask
                if history_mask is not None:
                    b = b.masked_fill(mask == 0, -1e9)
        
        return v  # (B, num_interests, capsule_dim)
    
    def squash(self, s):
        """
        Squash activation function for capsules
        
        @input:
            s: (B, num_interests, capsule_dim)
        @output:
            v: (B, num_interests, capsule_dim)
        """
        s_norm_sq = torch.sum(s ** 2, dim=-1, keepdim=True)  # (B, num_interests, 1)
        s_norm = torch.sqrt(s_norm_sq + 1e-9)  # (B, num_interests, 1)
        v = (s_norm_sq / (1 + s_norm_sq)) * (s / s_norm)
        return v

class DNN(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim = 1, dropout_rate = 0., do_batch_norm = True):
        super(DNN, self).__init__()
        self.in_dim = in_dim
        layers = []

        # hidden layers
        for hidden_dim in hidden_dims:
            linear_layer = nn.Linear(in_dim, hidden_dim)
            # torch.nn.init.xavier_uniform_(linear_layer.weight, gain=nn.init.calculate_gain('relu'))
            layers.append(linear_layer)
            in_dim = hidden_dim

            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            if do_batch_norm:
#                 layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.LayerNorm([hidden_dim]))

        # prediction layer
        last_layer = nn.Linear(in_dim, out_dim)
        layers.append(last_layer)
        # torch.nn.init.xavier_uniform_(last_layer.weight, gain=1.0)

        self.layers = nn.Sequential(*layers)
        
    def forward(self, inputs):
        """
        @input:
            `inputs`, [bsz, in_dim]
        @output:
            `logit`, [bsz, out_dim]
        """
        inputs = inputs.view(-1, self.in_dim)
        logit = self.layers(inputs)
        return logit
