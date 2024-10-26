from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load Your Model
model = BlipForConditionalGeneration.from_pretrained("Load_your_HuggingFace_Model")
processor = BlipProcessor.from_pretrained("Load_your_HuggingFace_Model")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
