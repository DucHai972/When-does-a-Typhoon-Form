import os
import gdown
from tensorflow.keras.models import load_model

def get_drive_url(model_name):
    model_drive_ids = {
        'resnet_t2.h5': '1FuE2F_5xpPm4F6URsLV67n145ewZQf6l',
        'resnet_t4.h5': '1Koyk1n_9OQ8SzqL-kc_z9Fvl8jM05Pj4',
        'resnet_t5.h5': '1-4w2KQ00UjdwMUC6brY-jt6BhkDL9d14',
        'resnet_t8.h5': '1djQd9O9mwy_Rmw0LalDvOlxSk_meb_zc',
        'resnet_t12.h5': '1cCIILDbs1gFFH4_pFXlKBxAy9viLaTQK',
        'resnet_t16.h5': '1w26TCcNO3zCV1NNKZLpcj4HgKWeeBXeq',
    }
    return f"https://drive.google.com/uc?id={model_drive_ids.get(model_name, '')}"

def download_model(drive_url, model_path):
    print(f"{model_path} not found. Downloading from {drive_url}...")
    gdown.download(drive_url, model_path, quiet=False)

def load_user_model(model_name):
    model_path = f"./model/{model_name}"
    model_dir = os.path.dirname(model_path)
    
    # Ensure the directory exists
    os.makedirs(model_dir, exist_ok=True)
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

