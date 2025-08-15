# tflite_evaluator.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import glob

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """
    Generates and displays a confusion matrix for the TFLite model evaluation.
    
    Args:
        y_true (np.array): True labels.
        y_pred (np.array): Predicted labels from the model.
        class_names (list): A list of strings for class names.
        model_name (str): The name of the model for the plot title.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix for {model_name} (INT8 TFLite)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

def evaluate_tflite_model(keras_model_path):
    """
    Loads a Keras model, converts it to an INT8 TFLite model, 
    evaluates its performance on the appropriate test set, and
    displays a confusion matrix.
    
    Args:
        keras_model_path (str): The file path to the saved Keras (.h5) model.
    """
    if not os.path.exists(keras_model_path):
        print(f"Error: Model file not found at {keras_model_path}")
        return

    # --- 1. Identify Dataset and Load Data ---
    model_name = os.path.basename(keras_model_path).replace('_model.h5', '')
    print(f"--- Processing Model: {model_name.upper()} ---")

    if 'mnist' in model_name and 'fashion' not in model_name:
        dataset_name = 'mnist'
        (x_train, _), (x_test, y_test) = keras.datasets.mnist.load_data()
        class_names = [str(i) for i in range(10)]
        target_shape = (28, 28, 1)
    elif 'fashion_mnist' in model_name:
        dataset_name = 'fashion_mnist'
        (x_train, _), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        target_shape = (28, 28, 1)
    elif 'cifar10' in model_name:
        dataset_name = 'cifar10'
        (x_train, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        target_shape = (32, 32, 3)
    else:
        print(f"Error: Could not determine dataset for model '{model_name}'. Skipping.")
        return

    # --- 2. Preprocess Data for Conversion and Evaluation ---
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # All models were trained on 32x32x3 images, so we must match that
    
    if dataset_name in ['mnist', 'fashion_mnist']:
        # Add a channel dimension, convert to 3 channels, and resize
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        # x_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_train))
        # x_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_test))
        # x_train = tf.image.resize(x_train, [target_shape[0], target_shape[1]]).numpy()
        # x_test = tf.image.resize(x_test, [target_shape[0], target_shape[1]]).numpy()

    # --- 3. Convert Keras Model to INT8 TFLite ---
    print("Loading Keras model...")
    model = keras.models.load_model(keras_model_path)

    # Representative dataset for quantization
    def representative_dataset_gen():
        for i in range(200): # Use 200 samples
            yield [x_train[i:i+1]]

    print("Converting model to INT8 TFLite with TF Ops...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT] # Optimizes for size and latency
    converter.representative_dataset = representative_dataset_gen
    # Ensure that if a TFLite op isn't available, it falls back to the TF op.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8, # Enable INT8 ops.
        tf.lite.OpsSet.SELECT_TF_OPS # Enable TF ops.
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model_quant = converter.convert()
    
    tflite_model_path = f"quantmodel/{model_name}_quant_int8.tflite"
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model_quant)
    print(f"TFLite model saved to: {tflite_model_path}")

    # --- 4. Evaluate the TFLite Model ---
    print("Evaluating TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = []
    for test_image in x_test:
        # Check if the input type is quantized, then scale data accordingly
        if input_details['dtype'] == np.int8:
            input_scale, input_zero_point = input_details['quantization']
            test_image_quantized = (test_image / input_scale) + input_zero_point
            test_image_quantized = np.expand_dims(test_image_quantized, axis=0).astype(input_details['dtype'])
            interpreter.set_tensor(input_details['index'], test_image_quantized)
        else: # Should not happen with our settings, but good practice
            interpreter.set_tensor(input_details['index'], np.expand_dims(test_image, axis=0))

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details['index'])
        
        # De-quantize the output if needed
        if output_details['dtype'] == np.int8:
            output_scale, output_zero_point = output_details['quantization']
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

        predicted_label = np.argmax(output_data)
        predictions.append(predicted_label)

    # --- 5. Calculate Accuracy and Show Confusion Matrix ---
    accuracy = np.sum(predictions == y_test.flatten()) / len(y_test)
    print(f"\nTFLite Model Accuracy: {accuracy:.4f}")

    plot_confusion_matrix(y_test.flatten(), np.array(predictions), class_names, model_name)
    print("-" * 50 + "\n")


if __name__ == '__main__':
    # Make sure the script is run from a directory containing the 'saved_models' folder
    model_files = glob.glob('saved_models/*_model.h5')
    
    if not model_files:
        print("No pre-trained Keras models (*_model.h5) found in the 'saved_models' directory.")
        print("Please run the training script first to generate the models.")
    else:
        for model_path in sorted(model_files):
            evaluate_tflite_model(model_path)

