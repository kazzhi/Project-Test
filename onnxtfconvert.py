import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import onnx
from onnx_tf.backend import prepare

# Load the ONNX model

print('Loaded success!!!')
onnx_model = onnx.load("model.onnx")

# Print the ONNX model's IR version and producer info
print(f"ONNX IR version: {onnx_model.ir_version}")
print(f"Producer name: {onnx_model.producer_name}")
print(f"Producer version: {onnx_model.producer_version}")

# Convert the ONNX model to TensorFlow
try:
    tf_rep = prepare(onnx_model)
    print("ONNX model successfully converted to TensorFlow.")
except Exception as e:
    print(f"Error during conversion: {e}")
    raise

# Export the TensorFlow model
tf_rep.export_graph("model_tf")
print("TensorFlow model saved to 'model_tf'.")