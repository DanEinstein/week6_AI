# =============================================================================
# TFLITE CONVERSION & BENCHMARKING SCRIPT
# =============================================================================

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import time
import shutil

# --- CONFIGURATION (Must match training script) ---
DATA_PATH = "waste_dataset"
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_H5_PATH = 'deployment_package/recyclables_mobilenet_0.35.h5'
TFLITE_OUTPUT_DIR = 'deployment_package'
# --------------------------------------------------

print("🚀 TFLite Conversion and Testing Initialized")
print("=" * 60)

# =============================================================================
# STEP 1: DATA PIPELINE (Reused for Evaluation)
# =============================================================================

def setup_validation_pipeline(data_path=DATA_PATH):
    """Sets up the validation data generator."""
    if not os.path.exists(data_path):
        print("⚠️ Data folder not found. Please run the training script first.")
        return None

    # Use a simple datagen for validation/testing
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    validation_generator = val_datagen.flow_from_directory(
        data_path,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False, # Crucial for accurate benchmarking
        seed=42
    )
    return validation_generator

# =============================================================================
# STEP 2: TENSORFLOW LITE CONVERSION (Float32 Baseline)
# =============================================================================

def convert_to_tflite_float32(keras_model_path):
    """Converts a Keras model to a Float32 TFLite model."""
    if not os.path.exists(keras_model_path):
        print(f"❌ Keras model not found at: {keras_model_path}")
        return None

    # Load the Keras model
    # Note: Custom objects may be required if using custom metrics/layers
    model = load_model(keras_model_path)

    print("\n🔄 Converting Keras Model to Float32 TFLite...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT] # Standard optimization, usually leads to float16/int8 mixes

    tflite_model = converter.convert()

    # Save the TFLite model
    tflite_path = os.path.join(TFLITE_OUTPUT_DIR, 'recyclables_mobilenet_float32.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    model_size = os.path.getsize(tflite_path) / 1024  # KB
    print(f"✅ Float32 TFLite model saved to: {tflite_path}")
    print(f"📦 Model Size: {model_size:.2f} KB")

    return tflite_model, tflite_path

# =============================================================================
# STEP 3: TFLITE BENCHMARKING (Accuracy and Latency)
# =============================================================================

def benchmark_tflite_performance(tflite_model, val_gen, samples_to_test=100):
    """Benchmarks TFLite model accuracy and inference speed."""
    print("\n⚡ Benchmarking TFLite Performance...")

    # Initialize interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    inference_times = []
    correct_predictions = 0
    total_samples = 0

    val_gen.reset()

    for batch_images, batch_labels in val_gen:
        for i in range(len(batch_images)):
            if total_samples >= samples_to_test:
                break

            # Prepare input tensor (must be float32)
            input_data = batch_images[i:i+1].astype(np.float32)

            # Time inference
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # Check accuracy (assuming binary classification: sigmoid output)
            true_label = batch_labels[i]
            pred_class = 1 if prediction[0][0] > 0.5 else 0

            if pred_class == true_label:
                correct_predictions += 1
            total_samples += 1

        if total_samples >= samples_to_test:
            break

    # Calculate metrics
    avg_inference_time_ms = np.mean(inference_times) * 1000
    std_inference_time_ms = np.std(inference_times) * 1000
    tflite_accuracy = correct_predictions / total_samples
    fps = 1000 / avg_inference_time_ms

    print(f"🎯 RESULTS (Samples Tested: {total_samples}):")
    print(f"    📊 Accuracy: {tflite_accuracy:.4f} ({tflite_accuracy*100:.2f}%)")
    print(f"    ⚡ Avg Inference Time: {avg_inference_time_ms:.2f} ± {std_inference_time_ms:.2f} ms")
    print(f"    📈 Throughput: {fps:.1f} FPS")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    # 1. Setup Data
    validation_generator = setup_validation_pipeline()
    if validation_generator is None:
        exit()

    # 2. Convert to TFLite (Float32)
    tflite_model_content, tflite_path = convert_to_tflite_float32(MODEL_H5_PATH)

    # 3. Benchmark
    if tflite_model_content:
        benchmark_tflite_performance(tflite_model_content, validation_generator, samples_to_test=200)

    print("\n✅ TFLite Conversion and Testing Complete.")