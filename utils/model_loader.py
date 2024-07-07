import os
import gdown
from tensorflow.keras.models import load_model

def get_drive_url(model_name):
    model_drive_ids = {
        'resnet_t2.h5': 'YOUR_FILE_ID_1',
        'resnet_t4.h5': 'YOUR_FILE_ID_2',
        'resnet_t5.h5': 'YOUR_FILE_ID_3',
        # Add more model names and their corresponding Drive IDs here
    }
    return f"https://drive.google.com/uc?id={model_drive_ids.get(model_name, '')}"

def download_model(drive_url, model_path):
    print(f"{model_path} not found. Downloading from {drive_url}...")
    gdown.download(drive_url, model_path, quiet=False)

def load_user_model(model_name):
    model_path = f"./model/{model_name}"
    drive_url = get_drive_url(model_name)
    
    if not os.path.isfile(model_path):
        if drive_url:
            download_model(drive_url, model_path)
        else:
            raise FileNotFoundError(f"Model {model_name} not found and no drive URL provided.")
    else:
        print(f"{model_path} already exists. Loading model...")
    
    model = load_model(model_path)
    return model

