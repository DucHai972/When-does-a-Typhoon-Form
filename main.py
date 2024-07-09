import argparse
from utils.model_loader import load_user_model
from utils.test_set_loader import load_test_set
from utils.label_loader import create_labels
from utils.data_preprocess import preprocess_data
from utils.metrics_calculator import calculate_and_save_metrics

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Select a model to use.")
parser.add_argument('--model', type=str, required=True, help='Name of the model file (e.g., resnet_t2.h5)')
args = parser.parse_args()

model_name = args.model

# Load the specified model
model = load_user_model(model_name)

# Load test set & preprocess
load_test_set()
normalized_test_set = preprocess_data()

# Load label correspondingly to model
labels = create_labels(model_name)
print("Model and test set loaded successfully.")

# Evaluate
predictions = model.predict(normalized_test_set)
predictions = (predictions > 0.5).astype(int)

calculate_and_save_metrics(labels, predictions)
