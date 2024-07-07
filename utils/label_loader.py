import numpy as np

def create_labels(model_name, num_samples=1000):
    """
    Create labels based on the model name.
    
    Parameters:
    model_name (str): The name of the model.
    num_samples (int): The total number of samples (length of y).
    
    Returns:
    np.ndarray: Array of labels (0s and 1s).
    """
    if model_name == 'resnet_t2.h5':
        num_ones = num_samples // 2
    elif model_name == 'resnet_t4.h5':
        num_ones = num_samples // 3
    elif model_name == 'resnet_t5.h5':
        num_ones = num_samples // 4
    else:
        num_ones = num_samples // 5
    
    num_zeros = num_samples - num_ones
    
    # Create the labels array
    labels = np.array([0] * num_zeros + [1] * num_ones)
    np.random.shuffle(labels)  # Shuffle to ensure randomness
    
    return labels

if __name__ == '__main__':
    # For testing purposes
    model_name = 'resnet_t2.h5'
    labels = create_labels(model_name)
    print(f"Labels for {model_name}:")
    print(labels)

