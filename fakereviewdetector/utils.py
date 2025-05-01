import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}:{seconds:02d}"

def save_results(metrics, time_taken, output_path):
    with open(output_path, 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")
        f.write(f"Time: {time_taken}\n")

def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro-Precision": precision_score(y_true, y_pred, average='macro'),
        "Macro-Recall": recall_score(y_true, y_pred, average='macro'),
        "Macro-F1 Score": f1_score(y_true, y_pred, average='macro'),
        "Weighted-Precision": precision_score(y_true, y_pred, average='weighted'),
        "Weighted-Recall": recall_score(y_true, y_pred, average='weighted'),
        "Weighted-F1 Score": f1_score(y_true, y_pred, average='weighted'),
    }