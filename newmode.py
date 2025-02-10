import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

data = "/data/"
IMG_SIZE = 64  # Resized image size
BATCH_SIZE = 32

def preprocess_image(image, label):
    image = tf.image.rgb_to_grayscale(image)  # Convert to grayscale
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize
    image = image / 255.0  # Normalize
    return image, label

# Load dataset (assuming it's in TensorFlow format)
full_train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data, "asl_alphabet_train"),
    image_size=(224, 224),
    batch_size=None,  # We will manually batch later
    shuffle=True
)

train_images = []
train_labels = []


for image, label in full_train_dataset:
    train_images.append(image.numpy())  # Convert TensorFlow tensor to NumPy
    train_labels.append(label.numpy())

train_images = np.array(train_images)
train_labels = np.array(train_labels)

train_images, _, train_labels, _ = train_test_split(
    train_images, train_labels, test_size=0.5, random_state=42, stratify=train_labels
)

X_train, X_val, Y_train, Y_val = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))

train_dataset = train_dataset.shuffle(len(train_images)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data, "asl_alphabet_test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_image)

# Shuffle and batch dataset
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE)


def build_lightweight_model(input_shape=(64, 64, 1), num_classes=29):  # Assuming grayscale input
    model = keras.Sequential([
        layers.Conv2D(16, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        layers.DepthwiseConv2D((3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.DepthwiseConv2D((3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.GlobalAveragePooling2D(),

        layers.Dense(29, activation="softmax")  # 26 letters + space + nothing + extra
    ])

    return model

model = build_lightweight_model()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()


datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_generator = datagen.flow(X_train, y_train, batch_size=32)
val_generator = datagen.flow(X_val, y_val, batch_size=32)

model.fit(train_generator, validation_data=val_generator, epochs=20)

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
