import time
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from fakereviewdetector.models import TransformerBiLSTMClassifier, TextDataset
from fakereviewdetector.utils import compute_metrics, format_time


def train_and_evaluate_bilstm(model_name, train_df, val_df, test_df, max_len, batch_size, epochs, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TransformerBiLSTMClassifier(model_name).to(device)

    train_dataset = TextDataset(train_df['text'].tolist(), train_df['class'].tolist(), tokenizer, max_len)
    test_dataset = TextDataset(test_df['text'].tolist(), test_df['class'].tolist(), tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
    end_time = time.time()

    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)
            preds.extend(predictions.cpu().numpy())
            true.extend(labels.cpu().numpy())

    metrics = compute_metrics(true, preds)
    duration = format_time(end_time - start_time)

    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "bilstm_model.pt"))
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "model_info.txt"), "w") as f:
        f.write(model_name)

    return metrics, duration
