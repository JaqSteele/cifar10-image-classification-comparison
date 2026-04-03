#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD


BASE_DIR = Path().resolve()
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    """
    Load and normalise the CIFAR-10 dataset.
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)


def build_cnn():
    """
    Build a simple baseline CNN for CIFAR-10 classification.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax")
    ])
    return model


def build_resnet50():
    """
    Build a ResNet-50-based classifier for CIFAR-10.
    Images are resized to 96x96 to better suit the architecture.
    """
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(96, 96, 3))
    )

    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    return model


def compile_model(model):
    """
    Compile the model with SGD optimiser.
    """
    model.compile(
        optimizer=SGD(learning_rate=0.01, momentum=0.9),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_and_evaluate(model, x_train, y_train, x_test, y_test, model_name, epochs=10, batch_size=32):
    """
    Train a model and return training history and evaluation metrics.
    """
    checkpoint_path = RESULTS_DIR / f"{model_name}_best.keras"

    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    y_pred_prob = model.predict(x_test, verbose=0)
    y_pred = tf.argmax(y_pred_prob, axis=1).numpy()
    y_true = y_test.flatten()

    metrics_dict = {
        "Model": model_name,
        "Accuracy": float(accuracy),
        "Precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1-score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    return history, metrics_dict


def plot_learning_curve(history, model_name):
    """
    Save a learning curve plot for a model.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{model_name} Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{model_name.lower()}_learning_curve.png")
    plt.close()


def plot_metric_comparison(results_df):
    """
    Save a bar chart comparing model metrics.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

    plt.figure(figsize=(10, 6))
    results_df.set_index("Model")[metrics].plot(kind="bar")
    plt.title("Model Performance Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png")
    plt.close()


def main():
    (x_train, y_train), (x_test, y_test) = load_data()

    # CNN uses original CIFAR-10 image size
    cnn_model = compile_model(build_cnn())
    history_cnn, metrics_cnn = train_and_evaluate(
        cnn_model, x_train, y_train, x_test, y_test, model_name="CNN"
    )
    plot_learning_curve(history_cnn, "CNN")

    # ResNet uses resized images
    x_train_resized = tf.image.resize(x_train, size=(96, 96)).numpy()
    x_test_resized = tf.image.resize(x_test, size=(96, 96)).numpy()

    resnet_model = compile_model(build_resnet50())
    history_resnet, metrics_resnet = train_and_evaluate(
        resnet_model, x_train_resized, y_train, x_test_resized, y_test, model_name="ResNet50"
    )
    plot_learning_curve(history_resnet, "ResNet50")

    results_df = pd.DataFrame([metrics_cnn, metrics_resnet])
    results_df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
    plot_metric_comparison(results_df)

    print("\nFinal Results:")
    print(results_df)


if __name__ == "__main__":
    main()


# In[ ]:




