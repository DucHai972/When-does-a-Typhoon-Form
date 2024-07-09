import numpy as np

def create_labels(model_name='resnet_t5.h5'):
    model_configs = {
        'resnet_t2.h5': (3, 34),
        'resnet_t4.h5': (5, 32),
        'resnet_t5.h5': (6, 31),
        'resnet_t8.h5': (9, 28),
        'resnet_t12.h5': (13, 24),
        'resnet_t16.h5': (17, 20),
        'cnn_t5.h5': (6, 31)
    }

    if model_name not in model_configs:
        raise ValueError(f"Unknown model name: {model_name}")

    num_ones, num_zeros = model_configs[model_name]

    # Create the labels array
    labels = np.array([1] * num_ones + [0] * num_zeros)
    return labels
