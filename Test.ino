#include "model_data.h" //Include C++ array


// TensorFlow Lite dependencies
#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

// Define a tensor arena size. You might need to adjust this based on your model requirements.
constexpr int tensor_arena_size = 8 * 1024;
byte tensor_arena[tensor_arena_size];

// Declare global variables for the TensorFlow Lite interpreter
static tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
static tflite::AllOpsResolver resolver;
static tflite::MicroInterpreter* interpreter = nullptr; // Use a pointer for the interpreter
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  while (!Serial) continue;

  // Set up logging (required for TensorFlow Lite Micro)
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure
  const tflite::Model* model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter, "Model schema version does not match TensorFlow Lite runtime.");
    return;
  }

  // Initialize the interpreter
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, tensor_arena_size, error_reporter);
  interpreter = &static_interpreter; // Point the global pointer to the newly created interpreter

  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Allocation of tensor memory failed.");
    return;
  }

  // Obtain pointers to the model's input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  float inputData[] = {-0.66f};

  // Ensure `input` and `interpreter` are not null
  if (input == nullptr || interpreter == nullptr) {
    Serial.println("Interpreter not initialized");
    return;
  }

  // Assign the input data to the tensor
  for (int i = 0; i < 1; ++i) { // Ensure this matches the size of your input tensor
    input->data.f[i] = inputData[i];
  }

  // Log the input data for verification
  Serial.print("Input: ");
  for (int i = 0; i < 1; ++i) {
    Serial.print(inputData[i], 4); // Print floating-point numbers with 4 digits after the decimal point
    Serial.print(" ");
  }
  Serial.println();

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Inference failed.");
    return;
  }

  // Log the model's output tensor values
  Serial.print("Output: ");
  for (int i = 0; i < output->bytes / sizeof(float); ++i) { // Adjust for the actual size and type of your output tensor
    Serial.print(output->data.f[i], 4); // Adjust this based on your model's output data type
    Serial.print(" ");
  }
  Serial.println();

  // Add a delay to slow down the loop for demonstration purposes
  delay(1000);
}
