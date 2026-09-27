import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    # Load hyperparameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["preprocess"]

    # Load raw data
    x_train_full = np.load("data/raw/x_train.npy")
    y_train_full = np.load("data/raw/y_train.npy")
    x_test = np.load("data/raw/x_test.npy")
    y_test = np.load("data/raw/y_test.npy")

    # Normalize pixel values to [0, 1]
    mean = x_train_full.mean()
    std = x_train_full.std()
    x_train_full = (x_train_full - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)

    # Split train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full,
        test_size=params["test_size"],
        random_state=params["seed"]
    )

    # Save processed arrays
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/train_images.npy", x_train)
    np.save("data/processed/train_labels.npy", y_train)
    np.save("data/processed/val_images.npy", x_val)
    np.save("data/processed/val_labels.npy", y_val)
    np.save("data/processed/test_images.npy", x_test)
    np.save("data/processed/test_labels.npy", y_test)

if __name__ == "__main__":
    main()