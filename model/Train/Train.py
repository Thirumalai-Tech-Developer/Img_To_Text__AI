import torch
from LoadModel import Load_Model as model
from LoadModel import device
from PreProcess import data_loader

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()

for epoch in range(3):  # Adjust the number of epochs as needed
    for batch in data_loader:
        inputs = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} completed with loss: {loss.item():.4f}")
