import torch
from LoadModel import Load_Model
from PreProcess import data_loader

# Initialize model instance and optimizer
model_instance = Load_Model("your_HuggingFace_Model")
optimizer = torch.optim.AdamW(model_instance.model.parameters(), lr=5e-5)
model_instance.model.train()

# Training loop
for epoch in range(3):  # Adjust the number of epochs as needed
    for batch in data_loader:
        inputs = {k: v.to(model_instance.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model_instance.model(**inputs)
        loss = outputs.loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} completed with loss: {loss.item():.4f}")
