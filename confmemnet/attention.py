import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MemoryAwareAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    d = config["embedding_dim"]
    self.q_proj = nn.Linear(d, d)
    self.k_proj_m = nn.Linear(d, d)
    self.v_proj_m = nn.Linear(d, d)
    self.k_proj_h = nn.Linear(d, d)
    self.v_proj_h = nn.Linear(d, d)
    self.k_proj_k = nn.Linear(d, d)
    self.v_proj_k = nn.Linear(d, d)
    self.confidence_net = nn.Sequential(nn.Linear(d, 1), nn.Sigmoid())
    self.dropout = nn.Dropout(0.1)

  def scaled_dot_attn(self, Q, K, V):
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(Q.size(-1))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V)

  def forward(self, x, M, H, K):
    Q = self.q_proj(x)
    z_m = self.scaled_dot_attn(Q, self.k_proj_m(M), self.v_proj_m(M))
    z_h = self.scaled_dot_attn(Q, self.k_proj_h(H), self.v_proj_h(H))
    z_k = self.scaled_dot_attn(Q, self.k_proj_k(K), self.v_proj_k(K))

    conf = self.confidence_net(x).squeeze(-1)
    out = conf.unsqueeze(-1) * z_m + ((1 - conf) * conf).unsqueeze(-1) * z_h + ((1 - conf) ** 2).unsqueeze(-1) * z_k
    return self.dropout(out), conf