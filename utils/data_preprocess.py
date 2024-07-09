import os
import numpy as np
import xarray as xr
import glob
import gdown
import joblib

def stack_variables(ds):
    multi_level_variables = ['ugrdprs', 'vgrdprs', 'vvelprs', 'tmpprs', 'hgtprs', 'rhprs']
    single_level_variables = ['tmpsfc', 'pressfc', 'landmask', 'hgttrp', 'tmptrp']
    isobaric_levels = [1000.0, 975.0, 950.0, 925.0, 900.0, 850.0, 800.0, 750.0, 700.0, 
                       650.0, 600.0, 550.0, 500.0, 450.0, 400.0, 350.0, 300.0, 250.0, 
                       200.0, 150.0, 100.0]
    
    stacked_vars = []

    # Stack the multi-isobaricInhPa variables
    for var_name in multi_level_variables:
        stacked_var = np.stack([ds[var_name].sel(isobaricInhPa=isobaric).values for isobaric in isobaric_levels], axis=-1)
        stacked_vars.append(stacked_var)

    # Stack the single-isobaricInhPa variables
    single_vars = np.stack([ds[var_name].values for var_name in single_level_variables], axis=-1)

    # Concatenate
    return np.concatenate(stacked_vars + [single_vars], axis=-1)


def stack_data(directory='./data/test_set/testdata_2019255/'):
    # List all NetCDF files in the directory
    nc_files = glob.glob(os.path.join(directory, '*.nc'))
    
    # Find the single positive file
    positive_file = None
    for f in nc_files:
        if 'POSITIVE' in os.path.basename(f):
            positive_file = f
            break

    # Separate negative files
    negative_files = [f for f in nc_files if 'NEGATIVE' in os.path.basename(f)]

    # Sort negative files based on the numeric timestamp
    negative_files.sort(key=lambda x: int(x.split('_')[4]))

    # Initialize an empty list to store the stacked arrays
    stacked_data = []

    # Process the positive file first
    ds = xr.open_dataset(positive_file)
    stacked_data.append(stack_variables(ds))

    # Process negative files in sorted order
    for file in negative_files:
        ds = xr.open_dataset(file)
        stacked_data.append(stack_variables(ds))

    # Convert list to numpy array
    stacked_data = np.array(stacked_data)
    
    return stacked_data


def sliding_window_aggregate(data, alpha=0.85):
    num_samples = len(data)
    aggregated_data = []
    for i in range(0, num_samples - 4):  # Adjusted range to aggregate every 5 samples
        aggregated_sample = None
        for j in range(5):
            if i + j < num_samples:
                if aggregated_sample is None:
                    aggregated_sample = data[i + j]
                else:
                    aggregated_sample = aggregated_sample + data[i + j] * (alpha ** j)
            else:
                break  # Stop aggregation if we reach the end of available samples
        aggregated_data.append(aggregated_sample)
    return np.array(aggregated_data)

def download_scaler():
    data_drive_id = '1fzwSUbpNrMylHrgbRW1bCBf2JY_90jKT'
    drive_url = f'https://drive.google.com/uc?id={data_drive_id}'
    output_path = './data/scaler.pkl'
    
    # Check if the file already exists
    if not os.path.exists(output_path):
        print(f"{output_path} not found. Downloading from {drive_url}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gdown.download(drive_url, output_path, quiet=False)
        print(f"File downloaded and saved as {output_path}")
    else:
        print(f"{output_path} already exists. Skipping download.")

def load_scaler():
    output_path = './data/scaler.pkl'
    with open(output_path, 'rb') as f:
        scaler = joblib.load(f)
    return scaler


def preprocess_data():
    # Download and load the scaler
    download_scaler()
    scaler = load_scaler()
    
    # Stack the data
    stacked_data = stack_data()
    squeezed_data = np.squeeze(stacked_data)
    aggregated_data = sliding_window_aggregate(squeezed_data)
    
    # Normalize the aggregated data
    aggregated_data_reshaped = aggregated_data.reshape(-1, aggregated_data.shape[-1])
    normalized_data = scaler.transform(aggregated_data_reshaped)
    normalized_data = normalized_data.reshape(aggregated_data.shape)
    
    return normalized_data
