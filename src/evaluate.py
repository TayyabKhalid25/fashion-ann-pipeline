import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def main():
    # Load model and test data
    model = load_model("models/model.h5")
    x_test = np.load("data/processed/test_images.npy")
    y_test = np.load("data/processed/test_labels.npy")

    # Compute metrics
    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    # Write metrics to root directory
    with open("metrics.json", "w") as f:
        json.dump({"test_loss": loss, "test_accuracy": acc}, f, indent=4)

    # Generate and save confusion matrix
    y_pred = np.argmax(model.predict(x_test), axis=-1)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig("confusion_matrix.png")

if __name__ == "__main__":
    main()