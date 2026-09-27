import os
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

def main():
    os.makedirs("data/raw", exist_ok=True)
    
    # Load Fashion-MNIST dataset directly via keras
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # Save raw arrays
    np.save("data/raw/x_train.npy", x_train)
    np.save("data/raw/y_train.npy", y_train)
    np.save("data/raw/x_test.npy", x_test)
    np.save("data/raw/y_test.npy", y_test)

if __name__ == "__main__":
    main()