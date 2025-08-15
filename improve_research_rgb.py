# main_training_script.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os

# Ensure the directory for saving models exists
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

def create_model(input_shape, num_classes):
    """
    Defines and returns the Keras Sequential model architecture.
    
    Args:
        input_shape (tuple): The shape of the input images (e.g., (28, 28, 1)).
        num_classes (int): The number of output classes.
        
    Returns:
        keras.Model: The compiled Keras model.
    """
    model = keras.Sequential([
        # Input Layer
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second Conv Block
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Classifier Head
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

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

def train_and_evaluate(dataset_name):
    """
    Loads a dataset, preprocesses it, trains the model, evaluates it,
    and displays a confusion matrix.
    
    Args:
        dataset_name (str): The name of the dataset to use ('mnist', 'fashion_mnist', or 'cifar10').
    """
    print(f"--- Processing Dataset: {dataset_name.upper()} ---")

    # 1. Load Data
    if dataset_name == 'mnist':
        (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
        class_names = [str(i) for i in range(10)]
        input_shape = (28, 28, 1)
    elif dataset_name == 'fashion_mnist':
        (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        input_shape = (28, 28, 1)
    elif dataset_name == 'cifar10':
        (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        input_shape = (32, 32, 3)
    else:
        raise ValueError("Dataset not supported. Choose 'mnist', 'fashion_mnist', or 'cifar10'.")

    # 2. Preprocess Data
    # Normalize pixel values to be between 0 and 1
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Reshape grayscale images to include the channel dimension
    if dataset_name in ['mnist', 'fashion_mnist']:
        x_train_full = np.expand_dims(x_train_full, -1)
        x_test = np.expand_dims(x_test, -1)

    # Convert labels to one-hot encoding
    num_classes = 10
    y_train_full_categorical = keras.utils.to_categorical(y_train_full, num_classes)
    y_test_categorical = keras.utils.to_categorical(y_test, num_classes)

    # 3. Create Validation Split (80:20)
    # We use the original y_train_full for stratification before it's one-hot encoded
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full_categorical, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Validation data shape: {x_val.shape}")
    print(f"Test data shape: {x_test.shape}")

    # 4. Create and Train the Model
    model = create_model(input_shape, num_classes)
    model.summary()

    print("\nStarting model training...")
    # Note: Training on CIFAR-10 will take longer than on MNIST
    history = model.fit(x_train, y_train,
                        batch_size=64,
                        epochs=20, # Increased epochs for the more complex dataset
                        validation_data=(x_val, y_val))
    print("Model training finished.")

    # 5. Evaluate the Model on the Test Set
    score = model.evaluate(x_test, y_test_categorical, verbose=0)
    print(f"\nTest loss: {score[0]:.4f}")
    print(f"Test accuracy: {score[1]:.4f}")

    # 6. Save the Model
    model_filename = f'saved_models/{dataset_name}_model.keras'
    model.save(model_filename)
    print(f"Model saved to {model_filename}")

    # 7. Generate and Display Confusion Matrix
    print("Generating confusion matrix...")
    y_pred_probs = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    # The original y_test labels are needed for the confusion matrix
    plot_confusion_matrix(y_test, y_pred_classes, class_names, dataset_name)
    print("-" * 50 + "\n")


if __name__ == '__main__':

    # Process the MNIST Digits dataset
    train_and_evaluate('mnist')
    
    # Process the Fashion-MNIST dataset
    train_and_evaluate('fashion_mnist')
    
    # Process the CIFAR-10 dataset
    train_and_evaluate('cifar10')
