from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

class Load_Model:
    def __init__(self, model_name="your_HuggingFace_Model"):
        # Load the model and processor from Hugging Face
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
