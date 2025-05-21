import time
import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from fakereviewdetector.utils import compute_metrics, format_time


def apply_pruning(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.utils.prune.l1_unstructured(module, name="weight", amount=amount)
            torch.nn.utils.prune.remove(module, "weight")


def train_and_evaluate_pruned(model_name, train_df, val_df, test_df, max_len, batch_size, epochs, pruning_amount, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_len)

    train_dataset = Dataset.from_pandas(train_df).rename_column("class", "labels").map(preprocess_function,
                                                                                       batched=True)
    val_dataset = Dataset.from_pandas(val_df).rename_column("class", "labels").map(preprocess_function, batched=True)
    test_dataset = Dataset.from_pandas(test_df).rename_column("class", "labels").map(preprocess_function, batched=True)

    for ds in [train_dataset, val_dataset, test_dataset]:
        ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    apply_pruning(model, amount=pruning_amount)

    training_args = TrainingArguments(
        output_dir="./pruned_model",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        logging_dir="./logs",
        report_to="none"
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset)

    start_time = time.time()
    trainer.train()
    end_time = time.time()

    predictions = trainer.predict(test_dataset)
    preds = torch.argmax(torch.tensor(predictions.predictions), dim=1).numpy()

    metrics = compute_metrics(test_df["class"], preds)
    duration = format_time(end_time - start_time)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return metrics, duration
