import time
from fakereviewdetector.simple_trainer import train_and_evaluate_simple
from fakereviewdetector.bilstm_trainer import train_and_evaluate_bilstm
from fakereviewdetector.pruning_trainer import train_and_evaluate_pruned


def train_and_evaluate(
    train_df,
    val_df,
    test_df,
    model_name,
    model_type=None,
    method="simple",
    max_len=128,
    batch_size=16,
    epochs=2,
    pruning_amount=0.3,  # используется только для pruning
    output_dir="saved_models/"
):
    """
    Точка запуска обучения. Метод: 'simple', 'bilstm', 'pruning'.
    """

    if method == "bilstm":
        return train_and_evaluate_bilstm(
            model_name=model_name,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            max_len=max_len,
            batch_size=batch_size,
            epochs=epochs,
            output_dir=output_dir
        )

    elif method == "pruning":
        return train_and_evaluate_pruned(
            model_name=model_name,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            max_len=max_len,
            batch_size=batch_size,
            epochs=epochs,
            pruning_amount=pruning_amount,
            output_dir=output_dir
        )

    elif method == "simple":
        if model_type is None:
            raise ValueError("model_type must be specified for method='simple'")
        return train_and_evaluate_simple(
            model_type=model_type,
            model_name=model_name,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            max_len=max_len,
            batch_size=batch_size,
            epochs=epochs,
            output_dir=output_dir
        )

    else:
        raise ValueError("Unknown method. Choose from: 'simple', 'bilstm', or 'pruning'.")
