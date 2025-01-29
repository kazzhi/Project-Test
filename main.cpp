#include "esp_log.h"
#include "esp_nn.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#define TFLITE_MODEL_FILE_PATH ""

static const char *TAG = "TFLM";

constexpr int kTensorArenaSize = 250 * 1024;  // Adjust based on memory availability
static uint8_t tensor_arena[kTensorArenaSize];

void setup_tflite_interpreter() {
    tflite::MicroErrorReporter micro_error_reporter;
    tflite::MicroMutableOpResolver<10> resolver; // Register ops


    // Find out if I can just load the MobileNetV3 or I need to manually create layers

    resolver.AddConv2D();       // Use ESP-NN optimized convolution
    resolver.AddDepthwiseConv2D();  // Use ESP-NN for depthwise conv
    resolver.AddFullyConnected();
    resolver.AddSoftmax();

    tflite::MicroInterpreter interpreter(
        tflite::GetModel(model_data), resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter
    );

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "Failed to allocate tensors!");
        return;
    }

    ESP_LOGI(TAG, "TFLite Micro Interpreter Initialized!");
}

static int8_t input_image_data[224 * 224 * 3];  // MobileNetV3 Input Size

void load_test_image() {
    FILE *img_file = fopen("", "rb");
    if (img_file == NULL) {
        ESP_LOGE(TAG, "Failed to open test image file");
        return;
    }
    fread(input_image_data, 1, sizeof(input_image_data), img_file);
    fclose(img_file);
}

void preprocess_image(uint8_t *raw_image, int8_t *quantized_image, float scale, int zero_point) {
    // Double check how to preprocess image
    for (int i = 0; i < 224 * 224 * 3; i++) {
        quantized_image[i] = (int8_t)((raw_image[i] - zero_point) * scale);
    }
}

void run_inference(tflite::MicroInterpreter *interpreter) {
    TfLiteTensor *input_tensor = interpreter->input(0);
    memcpy(input_tensor->data.int8, input_image_data, sizeof(input_image_data));

    int64_t start_time = esp_timer_get_time();
    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Failed to run inference");
        return;
    }
    int64_t end_time = esp_timer_get_time();
    ESP_LOGI(TAG, "Inference time: %lld ms", (end_time - start_time) / 1000);
}

void print_inference_results(tflite::MicroInterpreter *interpreter) {
    TfLiteTensor *output_tensor = interpreter->output(0);

    int8_t *output_data = output_tensor->data.int8;
    int num_classes = output_tensor->dims->data[1]; // Get class count

    ESP_LOGI(TAG, "Inference Results:");
    for (int i = 0; i < num_classes; i++) {
        float probability = (output_data[i] - output_tensor->params.zero_point) * output_tensor->params.scale;
        ESP_LOGI(TAG, "Class %d: Probability %.3f", i, probability);
    }
}


