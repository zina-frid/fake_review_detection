import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

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
        "Macro Precision": precision_score(y_true, y_pred, average='macro'),
        "Macro Recall": recall_score(y_true, y_pred, average='macro'),
        "Macro F1 Score": f1_score(y_true, y_pred, average='macro'),
        "Weighted Precision": precision_score(y_true, y_pred, average='weighted'),
        "Weighted Recall": recall_score(y_true, y_pred, average='weighted'),
        "Weighted F1 Score": f1_score(y_true, y_pred, average='weighted'),
    }


def predict_review(model_path, review_text):

    if "bilstm" in model_path.lower():
        # Load BiLSTM model
        from fakereviewdetector.models import TransformerBiLSTMClassifier

        with open(os.path.join(model_path, "model_info.txt"), "r") as f:
            original_model_name = f.read().strip()
    
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = TransformerBiLSTMClassifier(original_model_name)
        
        model.load_state_dict(torch.load(os.path.join(model_path, "bilstm_model.pt")))
        model.eval()
        
        # Токенизируем и предсказываем
        inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(inputs['input_ids'], inputs['attention_mask'])
        predicted = torch.argmax(logits, dim=1).item()
    else:
        # Загружаем токенизатор и модель
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        # Переводим модель в режим оценки (выключение dropout и других элементов)
        model.eval()

        # Токенизируем текст отзыва
        inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True)

        # Получаем логиты модели
        with torch.no_grad():
            logits = model(**inputs).logits
            
        # Получаем предсказание (класс с максимальным значением)
        predicted = torch.argmax(logits, dim=1).item()
    
    return predicted
