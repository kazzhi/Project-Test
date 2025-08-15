import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.keras import pruning

# Define constants
DATA_DIR = "data/"
IMG_SIZE = 64  # Resized image size
BATCH_SIZE = 32
NUM_CLASSES = 29  # 26 letters + space + nothing + extra

def preprocess_image(image, label):
    """Preprocess images: Convert to grayscale, resize, and normalize."""
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image, label

# Load dataset from directory
full_train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "asl_alphabet_train/"),
    image_size=(224, 224),  # Load images at 224x224 first
    batch_size=BATCH_SIZE,
    shuffle=True
).map(preprocess_image)  # Apply preprocessing

# Convert dataset to NumPy arrays

train_dataset = full_train_dataset.take(int(len(full_train_dataset) * 0.8))  # 80% train
val_dataset = full_train_dataset.skip(int(len(full_train_dataset) * 0.8))  # 20% validation

X_train = []
Y_train = []

for image, label in train_dataset:
    X_train.append(image.numpy())  # Convert TensorFlow tensor to NumPy
    Y_train.append(label.numpy())

X_train = np.concatenate(X_train, axis=0) # Concatenate the batches
y_train = np.concatenate(Y_train, axis=0)

X_val = []
y_val = []
for images, labels in val_dataset:
    X_val.append(images.numpy())
    y_val.append(labels.numpy())
X_val = np.concatenate(X_val, axis=0)
y_val = np.concatenate(y_val, axis=0)


# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

# Shuffle and batch datasets
train_dataset = train_dataset.shuffle(len(X_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

X_train = X_train.reshape(X_train.shape[0], IMG_SIZE, IMG_SIZE, 1)  # Use shape[0]
X_val = X_val.reshape(X_val.shape[0], IMG_SIZE, IMG_SIZE, 1)    


# Load and preprocess test dataset
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "asl_alphabet_test/"),
    image_size=(224, 224),
    batch_size=BATCH_SIZE,
    shuffle=False
).map(preprocess_image)  # Apply preprocessing

def build_lightweight_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES):
    """Builds a lightweight CNN model for image classification."""
    model = keras.Sequential([
        layers.Conv2D(8, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        layers.DepthwiseConv2D((3, 3), activation="relu", padding="same", depth_multiplier=1),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(16, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        layers.DepthwiseConv2D((3, 3), activation="relu", padding="same", depth_multiplier=1),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.DepthwiseConv2D((3, 3), activation="relu", padding="same", depth_multiplier=1),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

# Build and compile model
model = build_lightweight_model()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)



# Check shape after fix
print("X_train shape:", X_train.shape)  # Should be (num_samples, 64, 64, 1)
print("X_val shape:", X_val.shape) 

datagen.fit(X_train)

# Use ImageDataGenerator for augmentation
train_generator = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
val_generator = datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

# Train model
model.fit(train_generator, validation_data=val_generator, epochs=5)

# Evaluate on test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

model.save("asl_saved_model")
print("MODEL SAVED, MOVING ONTO PRUNING")


