import torch
import torch.nn as nn

def run_training(model, train_dl, valid_dl, config, optimizer, device, description="ConfMemNet"):
  loss_fn = nn.CrossEntropyLoss()
  for epoch in range(config["num_epochs"]):
    model.train()
    train_loss = 0
    for input_ids, targets in train_dl:
      input_ids, targets = input_ids.to(device), targets.to(device)
      logits = model(input_ids)
      loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      train_loss += loss.item()

    model.eval()
    valid_loss = 0
    with torch.no_grad():
      for input_ids, targets in valid_dl:
        input_ids, targets = input_ids.to(device), targets.to(device)
        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        valid_loss += loss.item()

    print(f"{description} Epoch {epoch+1} | Train Loss: {train_loss/len(train_dl):.4f} | Valid Loss: {valid_loss/len(valid_dl):.4f}")