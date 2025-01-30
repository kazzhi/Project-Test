import torch
import tensorflow as tf
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split


# Load MobileNetV3-Small pretrained on ImageNet
# Load face-mask detection dataset
# Split into training and test, 80:20
# Validation test
# Relative path: data/asl_alphabet_train/asl_alphabet_train
# Absolute path: /home/oefish/026prj/Project-Test/data/asl_alphabet_train/asl_alphabet_train
datapath = 'data/asl_alphabet_train/asl_alphabet_train' 
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

model = models.mobilenet_v3_small(pretrained=True)
model.to(device)

model.classifier[3] = nn.Linear(in_features=1024, out_features=29)
model.classifier[3].to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=datapath, transform=transform)

image, label = dataset[0]  # Access the first image
print("Image shape after transforms:", image.shape)

# Split dataset into train (80%) and validation (20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
print('train_size: ', train_size, ', val_size: ', val_size)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 3
best_val_loss = float("inf")
patience = 10
early_stop_counter = 0




for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)  # Move images to the correct device (CRUCIAL)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered")
            break

# model = torch.quantization.quantize_dynamic(
#     model, {nn.Linear}, dtype=torch.qint8
# )

torch.save(model.state_dict(), "trained_model.pth")
