import time
from simpletransformers.classification import ClassificationModel
from fakereviewdetector.utils import compute_metrics, format_time


def train_and_evaluate_simple(model_type, model_name, train_df, val_df, test_df, max_len, batch_size, epochs, output_dir):
    model = ClassificationModel(
        model_type,
        model_name,
        num_labels=2,
        use_cuda=True,
        args={
            'reprocess_input_data': True,
            'overwrite_output_dir': True,
            'num_train_epochs': epochs,
            'train_batch_size': batch_size,
            'learning_rate': 1e-5,
            'adam_epsilon': 1e-8,
            'max_seq_length': max_len,
            'dropout': 0.3,
            'evaluate_during_training': True,
            'evaluate_during_training_steps': 1000,
            'evaluate_during_training_verbose': True,
            'use_cuda': True,
            'silent': False
        }
    )

    start_time = time.time()
    print("Training the model")
    model.train_model(train_df, eval_df=val_df)
    end_time = time.time()

    print("Predictions")
    preds, _ = model.predict(test_df["text"].tolist())
    metrics = compute_metrics(test_df["class"], preds)
    duration = format_time(end_time - start_time)

    model.save_model(output_dir)

    return metrics, duration
