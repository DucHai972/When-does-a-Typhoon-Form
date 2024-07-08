import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from utils.model_loader import load_user_model
from utils.test_set_loader import load_test_set
from utils.label_loader import create_labels

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

labels = create_labels(model_name)
print("Model and test set loaded successfully.")

predictions = model.predict(test_set)
predictions = (predictions > 0.5).astype(int)

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

# Plotting metrics for label 0
metrics_0 = [precision_0, recall_0, f1_0]
labels_0 = ['Precision', 'Recall', 'F1 Score']

plt.figure(figsize=(8, 6))
plt.bar(labels_0, metrics_0, color='blue')
plt.ylim(0, 1)
plt.title('Performance Metrics for Label 0')
plt.ylabel('Score')
plt.savefig(f'{result_dir}/metrics_label_0.png')
plt.close()

# Plotting metrics for label 1
metrics_1 = [precision_1, recall_1, f1_1]
labels_1 = ['Precision', 'Recall', 'F1 Score']

plt.figure(figsize=(8, 6))
plt.bar(labels_1, metrics_1, color='green')
plt.ylim(0, 1)
plt.title('Performance Metrics for Label 1')
plt.ylabel('Score')
plt.savefig(f'{result_dir}/metrics_label_1.png')
plt.close()

print("Plots saved to the result directory")

# Plotting confusion matrix
cm = confusion_matrix(labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(f'{result_dir}/confusion_matrix.png')
plt.close()

print("Confusion matrix plot saved to the result directory")    
