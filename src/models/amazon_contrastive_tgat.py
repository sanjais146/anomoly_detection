import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalAmazonEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, use_time=True, use_decay=True):
        super().__init__()
        self.use_time = use_time
        self.use_decay = use_decay
        
        self.user_emb = nn.Linear(in_channels, hidden_channels)
        self.prod_emb = nn.Linear(in_channels, hidden_channels)
        
        if self.use_decay:
            self.raw_tau_user = nn.Parameter(torch.zeros(1))
            self.raw_tau_prod = nn.Parameter(torch.zeros(1))
            
        self.query_u = nn.Linear(hidden_channels, hidden_channels)
        self.key_p = nn.Linear(hidden_channels, hidden_channels)
        self.val_p = nn.Linear(hidden_channels, hidden_channels)
        
        self.query_p = nn.Linear(hidden_channels, hidden_channels)
        self.key_u = nn.Linear(hidden_channels, hidden_channels)
        self.val_u = nn.Linear(hidden_channels, hidden_channels)
        
    def get_tau(self, raw_tau):
        return F.softplus(raw_tau) + 1e-5
        
    def encode(self, u_idx, p_idx, t_target, edge_index, edge_attr, u_x, p_x):
        bs = len(u_idx)
        u_embeds = []
        p_embeds = []
        
        ux = self.user_emb(u_x)
        px = self.prod_emb(p_x)
        
        src, dst = edge_index
        edge_times = edge_attr[:, 0]
        
        if self.use_decay:
            tau_u = self.get_tau(self.raw_tau_user)
            tau_p = self.get_tau(self.raw_tau_prod)
            
        for i in range(bs):
            u = u_idx[i].item()
            p = p_idx[i].item()
            t = t_target[i].item()
            
            if self.use_time:
                # Strictly prior days!
                u_edges = (src == u) & (edge_times < t)
                p_edges = (dst == p) & (edge_times < t)
            else:
                # Static: just all edges (ignoring causality for the static ablation)
                u_edges = (src == u)
                p_edges = (dst == p)
                
            u_neighbors = dst[u_edges]
            u_t = edge_times[u_edges]
            
            if len(u_neighbors) > 0:
                q = self.query_u(ux[u])
                k = self.key_p(px[u_neighbors])
                v = self.val_p(px[u_neighbors])
                scores = (q * k).sum(dim=-1) / (q.size(-1)**0.5)
                attn = F.softmax(scores, dim=0)
                
                if self.use_decay and self.use_time:
                    delta_days = (t - u_t) / 86400.0
                    decay = torch.exp(-tau_u * delta_days)
                    agg = (attn.unsqueeze(1) * decay.unsqueeze(1) * v).sum(dim=0)
                else:
                    agg = (attn.unsqueeze(1) * v).sum(dim=0)
                u_embeds.append(ux[u] + agg)
            else:
                u_embeds.append(ux[u])
                
            p_neighbors = src[p_edges]
            p_t = edge_times[p_edges]
            
            if len(p_neighbors) > 0:
                q = self.query_p(px[p])
                k = self.key_u(ux[p_neighbors])
                v = self.val_u(ux[p_neighbors])
                scores = (q * k).sum(dim=-1) / (q.size(-1)**0.5)
                attn = F.softmax(scores, dim=0)
                
                if self.use_decay and self.use_time:
                    delta_days = (t - p_t) / 86400.0
                    decay = torch.exp(-tau_p * delta_days)
                    agg = (attn.unsqueeze(1) * decay.unsqueeze(1) * v).sum(dim=0)
                else:
                    agg = (attn.unsqueeze(1) * v).sum(dim=0)
                p_embeds.append(px[p] + agg)
            else:
                p_embeds.append(px[p])
                
        return torch.stack(u_embeds), torch.stack(p_embeds)


class HybridAmazonModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, use_time=True, use_decay=True, alpha=0.5):
        super().__init__()
        self.encoder = CausalAmazonEncoder(in_channels, hidden_channels, use_time, use_decay)
        self.alpha = alpha
        
    def forward(self, u_idx, p_idx, t_target, edge_index, edge_attr, u_x, p_x):
        # Generate embeddings
        u_emb, p_emb = self.encoder.encode(u_idx, p_idx, t_target, edge_index, edge_attr, u_x, p_x)
        # Score is dot product
        return (u_emb * p_emb).sum(dim=-1), u_emb, p_emb

    def compute_loss(self, pos_u, pos_p, neg_u, neg_p, t_target, edge_index, edge_attr, u_x, p_x):
        # Reconstruct positives
        pos_scores, pos_u_emb, pos_p_emb = self.forward(pos_u, pos_p, t_target, edge_index, edge_attr, u_x, p_x)
        recon_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
        
        # Contrastive with negatives (Margin loss)
        neg_scores, neg_u_emb, neg_p_emb = self.forward(neg_u, neg_p, t_target.repeat(len(neg_u)//len(pos_u)), edge_index, edge_attr, u_x, p_x)
        # InfoNCE / BPR Loss
        # log(sigmoid(pos - neg)). 
        # For simplicity, just BCE on negatives for recon, or margin for contrastive.
        # Let's decouple:
        # Recon = BCE(pos, 1) + BCE(neg, 0)
        recon_neg = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        full_recon = (recon_loss + recon_neg) / 2.0
        
        # Contrastive (Margin = 1.0)
        # Reshape negatives if multiple per positive
        k = len(neg_u) // len(pos_u)
        neg_scores_reshaped = neg_scores.view(len(pos_u), k)
        pos_scores_reshaped = pos_scores.unsqueeze(1)
        
        contrastive_loss = F.relu(1.0 - pos_scores_reshaped + neg_scores_reshaped).mean()
        
        hybrid_loss = self.alpha * full_recon + (1.0 - self.alpha) * contrastive_loss
        return hybrid_loss, full_recon, contrastive_loss
