import glob
from LoadModel import Load_Model
from PreProcess import CandlestickDataset
from torch.utils.data import DataLoader

# Initialize model and processor
model_instance = Load_Model("your_HuggingFace_Model")

# Load data from CSV files
csv_files = glob.glob("define_your_data_path/*.csv")
dataset = CandlestickDataset(csv_files, model_instance.processor)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
