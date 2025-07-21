import torch
import torch.nn as nn
from collections import Counter
from .utils import stop_tokens, clean_token

class MemoryManager(nn.Module):
  def __init__(self, config, tokenizer):
    super().__init__()
    self.salient_threshold = config["salient_threshold"]
    self.long_term_threshold = config["long_term_threshold"]
    self.salient_memory = None
    self.long_term_memory = None
    self.tokenizer = tokenizer
    self.token_freq = Counter()
    self.salient_tokens = []
    self.long_term_tokens = []
    self.stopword_penalty = config["stopword_penalty"]
    self.freq_penalty = config["freq_penalty"]

  def forward(self, x, confidence, input_ids):
    B, T = input_ids.shape
    confidence = confidence.clone()

    for b in range(B):
      toks = self.tokenizer.convert_ids_to_tokens(input_ids[b].tolist())
      for i in range(T):
        tok = clean_token(toks[i])
        self.token_freq[tok] += 1
        if tok in stop_tokens:
          confidence[b, i] *= self.stopword_penalty
        elif self.token_freq[tok] > 20:
          confidence[b, i] *= self.freq_penalty

    is_salient = confidence > self.salient_threshold
    is_important = confidence > self.long_term_threshold

    salient, long_term = [], []
    for b in range(B):
      if is_salient[b].any():
        salient.append(x[b][is_salient[b]].detach())
      if is_important[b].any():
        long_term.append(x[b][is_important[b]].detach())

    if salient:
      self.salient_memory = torch.cat(salient, dim=0).unsqueeze(0)
    if long_term:
      self.long_term_memory = torch.cat(long_term, dim=0).unsqueeze(0)

    return self.salient_memory, self.long_term_memory
