import argparse
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from utils.model_loader import load_user_model
from utils.test_set_loader import load_test_set

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Select a model to use.")
parser.add_argument('--model', type=str, required=True, help='Name of the model file (e.g., resnet_t2.h5)')
args = parser.parse_args()

model_name = args.model

# Load the specified model
model = load_user_model(model_name)

load_test_set()
test_set_path = './data/test_set.npy'
test_set = np.load(test_set_path)

num_samples = test_set.shape[0]  # Assuming test_set is a 2D or 3D array, adjust if necessary
labels = create_labels(model_name, num_samples=num_samples)

# Your additional code to use the model and test set...
print("Model and test set loaded successfully.")

predictions = model.predict(test_set)

# Calculate metrics
precision_0 = precision_score(labels, predictions, pos_label=0)
recall_0 = recall_score(labels, predictions, pos_label=0)
f1_0 = f1_score(labels, predictions, pos_label=0)

precision_1 = precision_score(labels, predictions, pos_label=1)
recall_1 = recall_score(labels, predictions, pos_label=1)
f1_1 = f1_score(labels, predictions, pos_label=1)

# Write metrics to a file
result_dir = './result'
os.makedirs(result_dir, exist_ok=True)
result_path = f'{result_dir}/result.txt'

with open(result_path, 'w') as f:
    f.write(f'Precision for label 0: {precision_0}\n')
    f.write(f'Recall for label 0: {recall_0}\n')
    f.write(f'F1 score for label 0: {f1_0}\n')
    f.write(f'Precision for label 1: {precision_1}\n')
    f.write(f'Recall for label 1: {recall_1}\n')
    f.write(f'F1 score for label 1: {f1_1}\n')

print(f"Metrics calculated and written to {result_path}")