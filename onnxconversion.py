import torch
import torch.nn as nn
import torchvision.models as models

# Load the quantized model
model = models.mobilenet_v3_small(pretrained=False)  # Reinitialize model architecture
model.classifier[3] = nn.Linear(in_features=1280, out_features=29)  # Match output classes
model.load_state_dict(torch.load("quantized_model.pth"))  # Load trained weights
model.eval()  # Set to evaluation mode

# Define dummy input (batch size 1, 3 color channels, 224x224 image)
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=11)
print("✅ Model successfully converted to ONNX format.")
