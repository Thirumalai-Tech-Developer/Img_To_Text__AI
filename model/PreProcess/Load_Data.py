import glob
from Change_RGB import CandlestickDataset , processor
from torch.utils.data import DataLoader
# Load Data

csv_files = glob.glob("define_your_data_path")
dataset = CandlestickDataset(csv_files, processor)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# for single file (optional)

#csv_files = "define_your_data_path"
#dataset = CandlestickDataset(csv_files, processor)
#data_loader = DataLoader(dataset, batch_size=4, shuffle=True)