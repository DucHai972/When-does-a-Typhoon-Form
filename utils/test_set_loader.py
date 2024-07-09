import gdown
import os
import patoolib

def load_test_set():
    data_drive_id = '14-XfVCVSpBdqNfirHdD31mjRFLVvkeRQ'
    drive_url = f'https://drive.google.com/uc?id={data_drive_id}'
    
    rar_path = './data/testdata_2019255.rar'
    extract_path = './data/test_set'

    # Check if the extracted data already exists
    if not os.path.exists(extract_path):
        print(f"{extract_path} not found. Downloading from {drive_url}...")
        os.makedirs(os.path.dirname(rar_path), exist_ok=True)
        
        # Download the .rar file
        gdown.download(drive_url, rar_path, quiet=False)
        print(f"File downloaded and saved as {rar_path}")
        
        # Extract the .rar file
        print(f"Extracting {rar_path} to {extract_path}...")
        patoolib.extract_archive(rar_path, outdir=extract_path)
        print(f"Extraction completed and saved at {extract_path}")
        
        # Remove .rar file
        os.remove(rar_path)
        print(f"Removed the .rar file: {rar_path}")
    else:
        print(f"{extract_path} already exists. Skipping download and extraction.")



