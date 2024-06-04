#include <TensorFlowLite.h>

#include <WiFi.h>
 #include "model.h"
//#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/kernels/micro_ops.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <cmath>
#include "image1.h" 
#include "image2.h"  

// TFLite model parameters
const tflite::Model* model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;
constexpr int kTensorArenaSize = 200 * 1024;
uint8_t* my_tensor_arena = nullptr;
uint8_t* model_data_psram = nullptr;

void processImage(const float* image_data);
void printEmbedding(float* embedding, int length);
float cosineSimilarity(float* vecA, float* vecB, int length);
void cleanEmbedding(float* embedding, int length);


void setup() {
    Serial.begin(115200);

    // Initialize PSRAM
    if (!psramFound()) {
        Serial.println("PSRAM not found");
        return;
    }

    // Allocate tensor arena in PSRAM
    my_tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);
    if (!my_tensor_arena) {
        Serial.println("Failed to allocate memory for tensor arena!");
        return;
    }

    // Allocate model data in PSRAM
    model_data_psram = (uint8_t*)ps_malloc(_content_model_tflite_len);
    if (!model_data_psram) {
        Serial.println("Failed to allocate memory for model data!");
        return;
    }
    memcpy(model_data_psram, _content_model_tflite, _content_model_tflite_len);

    // Initialize TFLite model
    model = tflite::GetModel(model_data_psram);
    static tflite::MicroErrorReporter micro_error_reporter;
    static tflite::AllOpsResolver resolver;

    static tflite::MicroInterpreter static_interpreter(model, resolver, my_tensor_arena, kTensorArenaSize, &micro_error_reporter);
    interpreter = &static_interpreter;

    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("AllocateTensors() failed");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    // Process the two images
    processImage(image1_data);
    float embedding1[64];
    memcpy(embedding1, output->data.f, 64 * sizeof(float));

    processImage(image2_data);
    float embedding2[64];
    memcpy(embedding2, output->data.f, 64 * sizeof(float));
    // Clean embedding
    cleanEmbedding(embedding1, 64);
    cleanEmbedding(embedding1, 64);
    // Calculate cosine similarity
    float similarity = cosineSimilarity(embedding1, embedding2, 64);
    Serial.println("Embedding 1:");
    printEmbedding(embedding1, 64);

    Serial.println("Embedding 2:");
    printEmbedding(embedding2, 64);
    Serial.print("Cosine Similarity: ");
    Serial.println(similarity);
}
void processImage(const float* image_data) {
    // Assuming image_data is 64x64x3 and already resized and normalized
    for (int i = 0; i < 64 * 64 * 3; i++) {
        input->data.f[i] = image_data[i];
    }
    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Invoke failed");
    }
}

void printEmbedding(float* embedding, int length) {
    for (int i = 0; i < length; i++) {
        Serial.print(embedding[i]);
        Serial.print(", ");
    }
    Serial.println();
}

float cosineSimilarity(float* vecA, float* vecB, int length) {
    float dotProduct = 0.0;
    float normA = 0.0;
    float normB = 0.0;
    for (int i = 0; i < length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    normA = sqrt(normA);
    normB = sqrt(normB);

    if (normA == 0 || normB == 0) {
        Serial.println("Warning: Zero norm encountered.");
        return NAN;  // Return NaN if any norm is zero
    }

    return dotProduct / (normA * normB);
}

void cleanEmbedding(float* embedding, int length) {
    for (int i = 0; i < length; i++) {
        if (isnan(embedding[i]) || isinf(embedding[i])) {
            embedding[i] = 0.0;
        }
    }
}

void loop() {
    // No need to do anything in loop
}
