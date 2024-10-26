from model.LoadModel import Load_Model
from model.PreProcess.Change_RGB import Change_RGB
from model.PreProcess.Load_Data import Load_Data
from model.Train.Train import Train

def main():
    # Initialize the model and processor
    model_instance = Load_Model("your_HuggingFace_Model")  # Load_Model is now an instance with model, processor, and device

    # Preprocess data (Change_RGB could be used on data if needed)
    csv_files = Load_Data()  # Load_Data should return the list of CSV paths
    dataset = Change_RGB(csv_files, model_instance.processor)  # Change_RGB processes the data

    # Initialize data loader
    data_loader = data_loader(dataset, batch_size=4, shuffle=True)

    # Train the model
    Train(model_instance.model, data_loader, model_instance.device)  # Pass the model, data_loader, and device to Train

if __name__ == "__main__":
    main()
