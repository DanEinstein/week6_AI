# =============================================================================
# LOCAL EDGE AI PIPELINE - MobileNetV2 + Full Integer Quantization
# Designed for local development and optimized for Raspberry Pi deployment.
# =============================================================================

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
import numpy as np
import os
import shutil
from PIL import Image
import time
import json
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DATA_PATH = "waste_dataset"
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 2  # Organic, Recyclable
ALPHA_VALUE = 0.35 # MobileNetV2 size factor (smaller is faster, less accurate)
# ---------------------

print("🚀 Local Edge AI Pipeline Initialized")
print("=" * 60)

# =============================================================================
# STEP 1: GUARANTEED DATASET SETUP (CIFAR-10 Proxy)
# =============================================================================

def create_local_dataset(data_path=DATA_PATH):
    """Creates a local, reproducible dataset structure using CIFAR-10."""
    if os.path.exists(data_path):
        shutil.rmtree(data_path)

    os.makedirs(f"{data_path}/organic", exist_ok=True)
    os.makedirs(f"{data_path}/recyclable", exist_ok=True)

    from tensorflow.keras.datasets import cifar10
    (x_train, y_train), _ = cifar10.load_data()

    # Class mappings (Vehicles=Recyclable, Animals=Organic)
    recyclable_classes = [0, 1, 8, 9]  # airplane, automobile, ship, truck
    organic_classes = [2, 3, 4, 5, 6, 7]  # bird, cat, deer, dog, frog, horse

    counts = {'recyclable': 0, 'organic': 0}
    max_per_class = 1000

    print("📥 Organizing CIFAR-10 into meaningful waste categories...")

    for i in range(len(x_train)):
        label = y_train[i][0]

        if label in recyclable_classes and counts['recyclable'] < max_per_class:
            class_name = 'recyclable'
        elif label in organic_classes and counts['organic'] < max_per_class:
            class_name = 'organic'
        else:
            continue

        img = Image.fromarray(x_train[i])
        # Resize images to target size (224x224) for MobileNet
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        img.save(f"{data_path}/{class_name}/{class_name}_{counts[class_name]:04d}.jpg")
        counts[class_name] += 1

        if counts['recyclable'] >= max_per_class and counts['organic'] >= max_per_class:
            break

    print(f"✅ Dataset created. Total images: {sum(counts.values())}")
    return data_path

# =============================================================================
# STEP 2: DATA PIPELINE
# =============================================================================

def setup_data_pipeline(data_path=DATA_PATH):
    """Sets up data generators for training and validation."""
    if not os.path.exists(data_path) or sum([len(os.listdir(f"{data_path}/{d}")) for d in os.listdir(data_path) if os.path.isdir(f"{data_path}/{d}")]) < 2000:
        data_path = create_local_dataset(data_path)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        data_path, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='binary',
        subset='training', shuffle=True, seed=42
    )

    validation_generator = val_datagen.flow_from_directory(
        data_path, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='binary',
        subset='validation', shuffle=False, seed=42
    )

    return train_generator, validation_generator

# =============================================================================
# STEP 3: MODEL ARCHITECTURE (MobileNetV2 with BN Fix)
# =============================================================================

def build_lightweight_model(num_classes=NUM_CLASSES):
    """Builds the MobileNetV2-based model for edge deployment."""

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3),
        alpha=ALPHA_VALUE
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))

    # Preprocess for MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # CRUCIAL BN FIX: Keep base model BN layers in inference mode (training=False)
    # This prevents corruption of ImageNet statistics by small batches.
    x = base_model(x, training=False)

    # Custom Classification Head
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # Output layer
    activation = 'sigmoid' if num_classes == 2 else 'softmax'
    units = 1 if num_classes == 2 else num_classes
    loss = 'binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy'

    outputs = Dense(units, activation=activation)(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model, base_model

# =============================================================================
# STEP 4: INTELLIGENT TRAINING STRATEGY
# =============================================================================

def train_model(model, base_model, train_gen, val_gen):
    """Runs Phase 1 and Phase 2 training."""

    # Callbacks for both phases
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1, min_delta=0.001),
    ]

    # --- Phase 1: Feature Extraction (Frozen Base) ---
    print("\n📚 Phase 1: Feature Extraction (Training head)...")
    model.fit(
        train_gen, epochs=15, validation_data=val_gen, callbacks=callbacks, verbose=1,
        workers=4, # Using 'workers' for local/Colab compatibility
    )

    # --- Phase 2: Fine-Tuning (Unfreeze Top Layers) ---
    print("\n🔧 Phase 2: Fine-tuning (Unfreezing last 30 layers)...")

    base_model.trainable = True
    for layer in base_model.layers[:-30]:  # Unfreeze only the last 30 layers
        layer.trainable = False

    # Recompile with very low learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss=model.loss,
        metrics=model.metrics
    )

    # More patient callbacks for fine-tuning
    fine_tune_callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.5)
    ]

    model.fit(
        train_gen, epochs=20, validation_data=val_gen, callbacks=fine_tune_callbacks,
        verbose=1, workers=4
    )

    return model

# =============================================================================
# STEP 5: TENSORFLOW LITE CONVERSION & QUANTIZATION (Int8)
# =============================================================================

def convert_and_quantize(model, train_generator):
    """Converts and quantizes the model to Int8 TFLite format."""
    print("\n🔄 Starting TFLite Conversion and Int8 Quantization...")

    # 1. Prepare Representative Dataset Generator
    def representative_data_gen():
        """Generator function to yield input data for quantization."""
        for input_value, _ in train_generator:
            # Yield one batch of float32 data
            yield [input_value.astype(np.float32)]

    # 2. Instantiate Converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 3. Apply Full Integer Quantization settings
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen

    # Ensure all operations can be quantized to integer (crucial for Raspberry Pi speed)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8 # Input tensor type for inference
    converter.inference_output_type = tf.uint8 # Output tensor type for inference

    # 4. Convert and Save
    tflite_model_int8 = converter.convert()

    output_dir = 'deployment_package'
    os.makedirs(output_dir, exist_ok=True)

    model_name = f'recyclables_mobilenet_{ALPHA_VALUE}_int8.tflite'
    tflite_path = os.path.join(output_dir, model_name)

    # Save the quantized model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model_int8)

    # Save unquantized H5 model for backup/comparison
    model.save(os.path.join(output_dir, f'recyclables_mobilenet_{ALPHA_VALUE}.h5'))

    # Save class indices for deployment
    with open(os.path.join(output_dir, 'class_indices.json'), 'w') as f:
        json.dump(train_generator.class_indices, f)

    print(f"✅ Quantization complete. Model saved to: {tflite_path}")
    print(f"📦 Model Size (Int8): {os.path.getsize(tflite_path) / 1024:.2f} KB")

    return tflite_path

# =============================================================================
# MAIN EXECUTION FLOW
# =============================================================================

if __name__ == '__main__':
    start_time = time.time()

    # 1. Setup Data Pipeline
    train_generator, validation_generator = setup_data_pipeline()

    # 2. Build and Compile Model
    model, base_model = build_lightweight_model(num_classes=NUM_CLASSES)

    # 3. Train Model
    trained_model = train_model(model, base_model, train_generator, validation_generator)

    # 4. Final Evaluation (before conversion)
    print("\n📊 Final H5 Model Evaluation:")
    loss, acc, prec, rec = trained_model.evaluate(validation_generator, verbose=0)
    print(f"Accuracy: {acc*100:.2f}% | Loss: {loss:.4f}")

    # 5. Convert and Quantize (Int8)
    tflite_model_path = convert_and_quantize(trained_model, train_generator)

    end_time = time.time()
    print(f"\n🎉 Pipeline Complete! Total Time: {end_time - start_time:.2f} seconds")

# EOF