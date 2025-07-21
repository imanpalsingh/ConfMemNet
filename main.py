import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from confmemnet.config import config
from confmemnet.data import AGNewsDataset, load_agnews
from confmemnet.model import ConfMemNet
from confmemnet.train import run_training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

train_data, valid_data = load_agnews(tokenizer, config["max_seq_len"])
train_ds = AGNewsDataset(train_data, tokenizer, config["max_seq_len"])
valid_ds = AGNewsDataset(valid_data, tokenizer, config["max_seq_len"])

train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=config["batch_size"])

model = ConfMemNet(config, tokenizer).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

run_training(model, train_dl, valid_dl, config, optimizer, device)
