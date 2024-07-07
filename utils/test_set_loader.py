import gdown

def load_test_set():
    # Google Drive URL of the .npy file
    drive_url = 'https://drive.google.com/uc?id=1UAsJ6LESJHLaw8cATerGZ7oAFAUp99Th'
    
    # Local path where the file will be saved
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
    download_test_set()

