"""
1D CNN Architecture, Compilation, Training, and Hyperparameter Experimentation Module
Author: Dilmith
Course: SE4050 - Deep Learning
Component: 1D CNN
"""

import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def build_cnn_model(
    vocab_size: int = 10000,
    embedding_dim: int = 128,
    filters: int = 128,
    kernel_size: int = 5,
    dense_units: int = 64,
    dropout_rate: float = 0.5,
    max_length: int = 200,
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Constructs and compiles a custom 1D CNN for binary text classification.

    Architecture Flow:
    Input review sequence (max_length integers)
    ↓
    Embedding (dense vector lookup table)
    ↓
    Conv1D (extracts local spatial/n-gram word features)
    ↓
    GlobalMaxPooling1D (extracts the strongest feature signal across the entire sequence)
    ↓
    Dense (non-linear feature combination)
    ↓
    Dropout (regularization to mitigate co-adaptation of neurons and overfitting)
    ↓
    Dense (Sigmoid activation for binary classification probability)

    Design Justifications:
    ---------------------
    - Sigmoid in Output Layer:
      Maps the final scalar logit to a continuous probability in the range [0, 1],
      representing P(Fake | Review). If P >= 0.5, the review is classified as Fake (1).
    - Binary Cross-Entropy Loss:
      The standard and statistically sound loss function for binary Bernoulli targets,
      heavily penalizing confident misclassifications via negative log-likelihood.
    - Adam Optimizer:
      Combines the benefits of AdaGrad (adaptive learning rates) and RMSProp (momentum),
      delivering fast convergence and stability for sparse text feature gradients.
    - ReLU Activation in Hidden Layers:
      Prevents gradient saturation/vanishing gradients common in sigmoid/tanh,
      provides computational efficiency, and induces sparse feature representations.
    - GlobalMaxPooling1D:
      Captures the most salient n-gram feature regardless of its position in the review,
      making the network shift-invariant and position-agnostic.

    Parameters:
        vocab_size (int): Size of the vocabulary dictionary.
        embedding_dim (int): Dimensionality of the dense word embedding.
        filters (int): Number of convolution filters (feature maps).
        kernel_size (int): Size of the 1D sliding convolution window (n-gram size).
        dense_units (int): Number of neurons in the penultimate dense layer.
        dropout_rate (float): Fraction of input units to drop during training.
        max_length (int): Fixed length of input sequences.
        learning_rate (float): Learning rate for Adam optimizer.

    Returns:
        tf.keras.Model: Compiled Keras Sequential 1D CNN model.
    """
    model = Sequential([
        Input(shape=(max_length,), dtype="int32", name="input_sequence"),
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name="embedding_layer"
        ),
        Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
            name="conv1d_feature_extractor"
        ),
        GlobalMaxPooling1D(name="global_max_pooling"),
        Dense(dense_units, activation="relu", name="dense_hidden"),
        Dropout(dropout_rate, name="dropout_regularization"),
        Dense(1, activation="sigmoid", name="output_sigmoid")
    ], name=f"1D_CNN_emb{embedding_dim}_f{filters}_k{kernel_size}")

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_cnn_model(
    model: tf.keras.Model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int = 10,
    batch_size: int = 32,
    patience: int = 2,
    verbose: int = 1
) -> tuple[tf.keras.callbacks.History, float, int]:
    """
    Trains the 1D CNN model using training and validation datasets.
    Measures wall-clock training time and applies EarlyStopping.

    Parameters:
        model (tf.keras.Model): Compiled CNN model.
        X_train: Padded training sequences.
        y_train: Training labels.
        X_val: Padded validation sequences.
        y_val: Validation labels.
        epochs (int): Maximum training epochs.
        batch_size (int): Mini-batch size.
        patience (int): Number of epochs without val_loss improvement before stopping.
        verbose (int): Verbosity mode.

    Returns:
        tuple: (history, training_time_seconds, actual_epochs_trained)
    """
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=verbose
    )

    start_time = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose
    )
    training_time = time.time() - start_time
    actual_epochs = len(history.history["loss"])

    return history, training_time, actual_epochs


def run_controlled_experiments(
    X_train,
    y_train,
    X_val,
    y_val,
    vocab_size: int = 10000,
    max_length: int = 200,
    max_epochs: int = 10,
    batch_size: int = 32,
    patience: int = 2,
    save_csv_path: str = "results/experiments/cnn_experiments.csv"
) -> tuple[pd.DataFrame, dict]:
    """
    Runs at least 3 controlled CNN hyperparameter experiments.
    Evaluates models on validation data only (never test data) to guide model selection.

    Experiment Matrix:
    - Experiment 1 (Compact / Trigram):
        Embedding=64, Filters=64, Kernel size=3, Dense=64, Dropout=0.3
    - Experiment 2 (Baseline / 5-gram):
        Embedding=128, Filters=128, Kernel size=5, Dense=64, Dropout=0.5
    - Experiment 3 (High Capacity / 5-gram):
        Embedding=128, Filters=256, Kernel size=5, Dense=128, Dropout=0.5

    Returns:
        tuple: (results_df, trained_models_dict)
    """
    experiment_configs = [
        {
            "experiment_id": "CNN_Exp_1_Compact",
            "description": "Compact representation (Trigram, 64-dim embedding)",
            "embedding_dim": 64,
            "filters": 64,
            "kernel_size": 3,
            "dense_units": 64,
            "dropout": 0.3,
        },
        {
            "experiment_id": "CNN_Exp_2_Baseline",
            "description": "Baseline representation (5-gram, 128-dim embedding)",
            "embedding_dim": 128,
            "filters": 128,
            "kernel_size": 5,
            "dense_units": 64,
            "dropout": 0.5,
        },
        {
            "experiment_id": "CNN_Exp_3_HighCapacity",
            "description": "High capacity feature maps (256 filters, 128 dense units)",
            "embedding_dim": 128,
            "filters": 256,
            "kernel_size": 5,
            "dense_units": 128,
            "dropout": 0.5,
        },
    ]

    results = []
    models_dict = {}

    for config in experiment_configs:
        exp_id = config["experiment_id"]
        print(f"\n=======================================================")
        print(f"Executing Controlled Experiment: {exp_id}")
        print(f"Embedding: {config['embedding_dim']} | Filters: {config['filters']} | "
              f"Kernel: {config['kernel_size']} | Dense: {config['dense_units']} | Dropout: {config['dropout']}")
        print(f"=======================================================")

        # Build model
        model = build_cnn_model(
            vocab_size=vocab_size,
            embedding_dim=config["embedding_dim"],
            filters=config["filters"],
            kernel_size=config["kernel_size"],
            dense_units=config["dense_units"],
            dropout_rate=config["dropout"],
            max_length=max_length
        )

        param_count = model.count_params()

        # Train model with EarlyStopping on validation loss
        history, train_time, actual_epochs = train_cnn_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=max_epochs,
            batch_size=batch_size,
            patience=patience,
            verbose=1
        )

        # Retrieve best validation performance
        val_losses = history.history["val_loss"]
        val_accuracies = history.history["val_accuracy"]
        best_epoch_idx = int(val_losses.index(min(val_losses)))

        best_val_loss = val_losses[best_epoch_idx]
        best_val_acc = val_accuracies[best_epoch_idx]
        train_loss_at_best = history.history["loss"][best_epoch_idx]
        train_acc_at_best = history.history["accuracy"][best_epoch_idx]

        results.append({
            "experiment_id": exp_id,
            "description": config["description"],
            "embedding_dim": config["embedding_dim"],
            "filters": config["filters"],
            "kernel_size": config["kernel_size"],
            "dense_units": config["dense_units"],
            "dropout": config["dropout"],
            "optimizer": "Adam",
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "actual_epochs": actual_epochs,
            "best_epoch": best_epoch_idx + 1,
            "train_accuracy": round(float(train_acc_at_best), 4),
            "val_accuracy": round(float(best_val_acc), 4),
            "train_loss": round(float(train_loss_at_best), 4),
            "val_loss": round(float(best_val_loss), 4),
            "generalization_gap": round(float(train_acc_at_best - best_val_acc), 4),
            "training_time_sec": round(float(train_time), 2),
            "param_count": int(param_count),
        })

        models_dict[exp_id] = {
            "model": model,
            "history": history,
            "config": config,
            "best_epoch": best_epoch_idx + 1
        }

    results_df = pd.DataFrame(results)

    if save_csv_path:
        import os
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        results_df.to_csv(save_csv_path, index=False)
        print(f"\n[Saved] Experiment comparison saved to: {save_csv_path}")

    return results_df, models_dict
