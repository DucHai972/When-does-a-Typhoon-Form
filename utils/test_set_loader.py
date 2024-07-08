import gdown
import os

def load_test_set():
    model_drive_id = '1UAsJ6LESJHLaw8cATerGZ7oAFAUp99Th'
    drive_url = f'https://drive.google.com/uc?id={model_drive_id}'
    
    output_path = './data/test_set.npy'
    
    # Check if the file already exists
    if not os.path.exists(output_path):
        print(f"{output_path} not found. Downloading from {drive_url}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gdown.download(drive_url, output_path, quiet=False)
        print(f"File downloaded and saved as {output_path}")
    else:
        print(f"{output_path} already exists. Skipping download.")

if __name__ == '__main__':
    load_test_set()

