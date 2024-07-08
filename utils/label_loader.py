import numpy as np

def create_labels(model_name='resnet_t5.h5'):
    model_configs = {
        'resnet_t2.h5': (236, 2105),
        'resnet_t4.h5': (390, 1951),
        'resnet_t5.h5': (467, 1874),
        'resnet_t8.h5': (697, 1644),
        'resnet_t12.h5': (988, 1353),
        'resnet_t16.h5': (1261, 1080)
    }

    if model_name not in model_configs:
        raise ValueError(f"Unknown model name: {model_name}")

    num_ones, num_zeros = model_configs[model_name]

    # Create the labels array
    labels = np.array([0] * num_zeros + [1] * num_ones)
    print(num_zeros, num_ones)
    return labels

def main():
    #Testing
    model_name = 'resnet_t5.h5'
    labels = create_labels(model_name)
    print(f"Labels for {model_name}:")
    print(labels)

if __name__ == '__main__':
    main()
