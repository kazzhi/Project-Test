import keras
import tensorflow as tf
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim


# Load MobileNetV3-Small pretrained on ImageNet
output = 10
datapath = ''

model = models.mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Linear(in_features=1280, out_features=output)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = datasets.ImageFolder(root=datapath, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Adjust based on your problem
              metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monit    or validation loss
    patience=10,         # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore the best weights when stopping
)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=val_data,
    callbacks=[early_stopping]
)

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)