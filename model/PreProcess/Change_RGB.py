from torch.utils.data import Dataset
import pandas as pd
from PIL import Image 
import os

# Custom Dataset
class CandlestickDataset(Dataset):
    def __init__(self, csv_files, processor):
        # Combine all CSV files into a single DataFrame
        self.data = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['Input']
        label = row['Output']  # Corresponding label

        # Check if the image file exists
        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_path} not found. Skipping entry.")
            return self.__getitem__((idx + 1) % len(self.data))  # Move to the next item if missing

        # Load image and convert it to RGB format
        image = Image.open(image_path).convert("RGB")
        
        # Prepare inputs for the model
        inputs = self.processor(images=image, text=label, return_tensors="pt", padding=True)
        inputs['labels'] = self.processor(text=[label], return_tensors="pt")["input_ids"]
        
        return {k: v.squeeze() for k, v in inputs.items()}
