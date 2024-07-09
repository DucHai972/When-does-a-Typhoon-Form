import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def calculate_and_save_metrics(labels, predictions, result_dir='./result'):
    # Calculate metrics
    precision_0 = precision_score(labels, predictions, pos_label=0)
    recall_0 = recall_score(labels, predictions, pos_label=0)
    f1_0 = f1_score(labels, predictions, pos_label=0)

    precision_1 = precision_score(labels, predictions, pos_label=1)
    recall_1 = recall_score(labels, predictions, pos_label=1)
    f1_1 = f1_score(labels, predictions, pos_label=1)

    # Write metrics to a file
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

    # Plotting metrics
    metrics_0 = [round(precision_0, 2), round(recall_0, 2), round(f1_0, 2)]
    metrics_1 = [round(precision_1, 2), round(recall_1, 2), round(f1_1, 2)]
    labels_ = ['Precision', 'Recall', 'F1 Score']
    x = np.arange(len(labels_))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, metrics_0, width, label='Label 0', color='blue')
    rects2 = ax.bar(x + width/2, metrics_1, width, label='Label 1', color='green')

    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics for Labels 0 and 1')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    plt.savefig(f'{result_dir}/metrics.png')
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

