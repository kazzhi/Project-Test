import torch
import torch.nn as nn
import torchvision.models as models


model = models.mobilenet_v3_small(pretrained=False) # Same architecture
model.classifier[3] = nn.Linear(in_features=1024, out_features=29) # Same classifier
model.load_state_dict(torch.load("trained_model.pth")) # Load the weights
model.eval()

# Dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX (no quantization here!)
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=11)  # Or your opset

print("✅ Model successfully converted to ONNX format (non-quantized).")
