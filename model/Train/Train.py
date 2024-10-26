import torch
from LoadModel import Load_Model
from PreProcess import data_loader

# Training loop
optimizer = torch.optim.AdamW(Load_Model.model.parameters(), lr=5e-5)
Load_Model.model.train()

for epoch in range(3):  # Adjust the number of epochs as needed
    for batch in data_loader:
        inputs = {k: v.to(Load_Model.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = Load_Model.model(**inputs)
        loss = outputs.loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} completed with loss: {loss.item():.4f}")
