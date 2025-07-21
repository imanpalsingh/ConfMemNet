import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class AGNewsDataset(Dataset):
  def __init__(self, data, tokenizer, max_len):
    self.tokenizer = tokenizer
    self.data = data
    self.max_len = max_len

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    tokens = torch.tensor(self.data[idx]["input_ids"])
    targets = tokens.clone()
    targets[:-1] = tokens[1:]
    targets[-1] = self.tokenizer.eos_token_id
    return tokens, targets

def load_agnews(tokenizer, max_len):
  dataset = load_dataset("ag_news", split="train")
  train_data = dataset.select(range(5000))
  valid_data = dataset.select(range(5000, 6000))

  def tokenize_fn(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=max_len)

  tokenized_train = train_data.map(tokenize_fn)
  tokenized_valid = valid_data.map(tokenize_fn)

  return tokenized_train, tokenized_valid
