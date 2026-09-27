import os
import csv
import yaml
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def main():
    # Load hyperparameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["train"]

    # Load processed data
    x_train = np.load("data/processed/train_images.npy")
    y_train = np.load("data/processed/train_labels.npy")
    x_val = np.load("data/processed/val_images.npy")
    y_val = np.load("data/processed/val_labels.npy")

    # Build the ANN
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(params["dense_units"], activation='relu'),
        Dropout(params["dropout_rate"]),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    optimizer = Adam(learning_rate=params["learning_rate"])
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    os.makedirs("models", exist_ok=True)
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"]
    )

    # Save the trained model and training history
    model.save("models/model.h5")
    
    history_dict = history.history
    with open("models/history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(history_dict.keys())
        writer.writerows(zip(*history_dict.values()))

if __name__ == "__main__":
    main()