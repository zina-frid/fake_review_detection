import torch
import torch.nn as nn
from transformers import AutoModel
from torch.utils.data import Dataset


class TransformerBiLSTMClassifier(nn.Module):
    def __init__(self, model_name, hidden_dim=256, num_labels=2, dropout=0.3):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.lstm = nn.LSTM(input_size=self.transformer.config.hidden_size,
                            hidden_size=hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        lstm_output, _ = self.lstm(outputs.last_hidden_state)
        pooled_output = lstm_output[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx])
        }