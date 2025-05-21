import time
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from fakereviewdetector.utils import compute_metrics, format_time


def train_and_evaluate_simple(model_name, train_df, val_df, test_df, max_len, batch_size, epochs, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_len)

    train_dataset = Dataset.from_pandas(train_df).rename_column("class", "labels").map(preprocess_function, batched=True)
    val_dataset = Dataset.from_pandas(val_df).rename_column("class", "labels").map(preprocess_function, batched=True)
    test_dataset = Dataset.from_pandas(test_df).rename_column("class", "labels").map(preprocess_function, batched=True)

    for ds in [train_dataset, val_dataset, test_dataset]:
        ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    start_time = time.time()
    trainer.train()
    end_time = time.time()

    predictions = trainer.predict(test_dataset)
    preds = torch.argmax(torch.tensor(predictions.predictions), dim=1).numpy()

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = compute_metrics(test_df["class"], preds)
    duration = format_time(end_time - start_time)

    return metrics, duration
