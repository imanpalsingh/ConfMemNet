import torch
import torch.nn as nn
from .attention import MemoryAwareAttention
from .memory import MemoryManager

class TokenEncoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.token_embedding = nn.Embedding(config["vocab_size"], config["embedding_dim"])
    self.pos_embedding = nn.Parameter(torch.randn(1, config["max_seq_len"], config["embedding_dim"]))
    self.dropout = nn.Dropout(0.1)

  def forward(self, input_ids):
    tok_embed = self.token_embedding(input_ids)
    pos_embed = self.pos_embedding[:, :tok_embed.size(1), :]
    return self.dropout(tok_embed + pos_embed)

class TransformerBlock(nn.Module):
  def __init__(self, config, tokenizer):
    super().__init__()
    self.attn = MemoryAwareAttention(config)
    self.ffn = nn.Sequential(
      nn.Linear(config["embedding_dim"], config["ffn_dim"]),
      nn.GELU(),
      nn.Linear(config["ffn_dim"], config["embedding_dim"])
    )
    self.norm1 = nn.LayerNorm(config["embedding_dim"])
    self.norm2 = nn.LayerNorm(config["embedding_dim"])
    self.mem_mgr = MemoryManager(config, tokenizer)

  def forward(self, x, H, K, input_ids):
    M = x
    attn_out, conf = self.attn(x, M, H, K)
    H_new, K_new = self.mem_mgr(attn_out, conf, input_ids)
    x = x + self.norm1(attn_out)
    x = x + self.norm2(self.ffn(x))
    return x, H_new, K_new

class ConfMemNet(nn.Module):
  def __init__(self, config, tokenizer):
    super().__init__()
    self.encoder = TokenEncoder(config)
    self.blocks = nn.ModuleList([TransformerBlock(config, tokenizer) for _ in range(config["num_layers"])])
    self.head = nn.Linear(config["embedding_dim"], config["vocab_size"])
    self.tokenizer = tokenizer

  def forward(self, input_ids):
    x = self.encoder(input_ids)
    H, K = x, x
    for block in self.blocks:
      x, H, K = block(x, H, K, input_ids)
    return self.head(x)
