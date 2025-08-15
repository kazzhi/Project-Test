import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
from tensorflow.keras.models import load_model


def plot_confusion_matrix(y_true, y_pred, class_names, dataset_name):
    """
    Generates and displays a confusion matrix.
    
    Args:
        y_true (np.array): True labels.
        y_pred (np.array): Predicted labels.
        class_names (list): A list of strings representing the class names.
        dataset_name (str): The name of the dataset for the plot title.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

num_classes = 10
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
input_shape = (32, 32, 3)
y_train_full_categorical = keras.utils.to_categorical(y_train_full, num_classes)
y_test_categorical = keras.utils.to_categorical(y_test, num_classes)
x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

model = load_model('saved_models/cifar10_model.keras')
score = model.evaluate(x_test, y_test_categorical, verbose=0)
print("Generating confusion matrix...")
y_pred_probs = model.predict(x_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# The original y_test labels are needed for the confusion matrix
plot_confusion_matrix(y_test, y_pred_classes, class_names, dataset_name)
print("-" * 50 + "\n")