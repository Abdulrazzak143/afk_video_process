"""
Train a structural-inspection image classifier (crack / leakage / normal)
with MobileNetV2 transfer learning, then export a quantized TFLite model.

Usage (from the project root):
    pip install tensorflow
    python train_model.py

Expected layout:
    dataset/
        crack/
        leakage/
        normal/
"""

import os
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Paths and training hyperparameters
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
ASSETS_DIR = os.path.join(SCRIPT_DIR, "assets")
TFLITE_PATH = os.path.join(ASSETS_DIR, "crack_leakage_model.tflite")
LABELS_PATH = os.path.join(ASSETS_DIR, "labels.txt")

# Required class folders (must match subdirectory names under dataset/)
EXPECTED_CLASSES = ["crack", "leakage", "normal"]

IMG_SIZE = (224, 224)  # MobileNetV2 native input size
BATCH_SIZE = 32
SEED = 42

# Two-phase transfer learning:
#   1) train only the new classification head
#   2) unfreeze the last few base-model layers and fine-tune at a lower LR
HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 10
HEAD_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_AT = 100  # unfreeze MobileNetV2 layers from this index onward


def require_dataset():
    """Fail fast if dataset/ or any class folder is missing."""
    if not os.path.isdir(DATASET_DIR):
        print(f"ERROR: Dataset folder not found: {DATASET_DIR}")
        print("Create dataset/crack, dataset/leakage, and dataset/normal with images.")
        sys.exit(1)

    missing = [
        name
        for name in EXPECTED_CLASSES
        if not os.path.isdir(os.path.join(DATASET_DIR, name))
    ]
    if missing:
        print(f"ERROR: Missing class folders under dataset/: {missing}")
        sys.exit(1)

    # Keras loads class names alphabetically; keep labels.txt in that order.
    class_names = sorted(
        name
        for name in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, name))
    )
    if class_names != EXPECTED_CLASSES:
        print(
            "ERROR: dataset/ must contain exactly these folders "
            f"(alphabetical): {EXPECTED_CLASSES}. Found: {class_names}"
        )
        sys.exit(1)

    return class_names


def load_datasets():
    """
    Load 224x224 RGB images from dataset/ with an 80/20 train/validation split.

    Images stay in the 0–255 range. The Keras model’s first layer maps them
    to MobileNetV2’s expected [-1, 1] range, which also matches on-device
    inference that feeds raw RGB bytes.
    """
    common = dict(
        directory=DATASET_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        label_mode="categorical",
        seed=SEED,
        validation_split=0.2,
    )
    train_ds = keras.utils.image_dataset_from_directory(
        subset="training",
        shuffle=True,
        **common,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        subset="validation",
        shuffle=False,
        **common,
    )

    # Prefetch so the GPU/CPU is not stalled waiting on disk I/O.
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds


def build_model(num_classes):
    """
    MobileNetV2 backbone (ImageNet weights) + a small classification head.

    Input: float32 [batch, 224, 224, 3] in 0–255 RGB.
    Output: softmax probabilities over crack / leakage / normal.
    """
    inputs = keras.Input(shape=IMG_SIZE + (3,), name="image")

    # Light augmentation only on the training path (inactive at inference).
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.08)(x)

    # MobileNetV2 preprocess: pixel = pixel / 127.5 - 1  →  [-1, 1]
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")(x)

    base = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # freeze convolutional feature extractor at first

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="crack_leakage_mobilenetv2")
    model.compile(
        optimizer=keras.optimizers.Adam(HEAD_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def train(model, base, train_ds, val_ds):
    """Phase 1: train the head. Phase 2: fine-tune the top of MobileNetV2."""
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
        ),
    ]

    print("\n=== Phase 1: training classification head (base frozen) ===")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=HEAD_EPOCHS,
        callbacks=callbacks,
    )

    # Unfreeze later MobileNetV2 blocks for a short, low-LR fine-tune.
    base.trainable = True
    for layer in base.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(FINE_TUNE_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\n=== Phase 2: fine-tuning upper MobileNetV2 layers ===")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks,
    )
    return model


def export_labels(class_names):
    """Write one class name per line so the Flutter app can map output indices."""
    os.makedirs(ASSETS_DIR, exist_ok=True)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(class_names) + "\n")
    print(f"Wrote labels: {LABELS_PATH}")


def export_quantized_tflite(model, train_ds):
    """
    Post-training full-integer quantization.

    The converter uses a representative sample of training images (0–255 RGB)
    to calibrate activations. Input and output tensors are uint8 so on-device
    inference can pass raw RGB bytes and read quantized class scores.
    """
    os.makedirs(ASSETS_DIR, exist_ok=True)

    def representative_dataset():
        # ~100 batches is more than enough for calibration; cap for speed.
        for i, (images, _) in enumerate(train_ds):
            if i >= 100:
                break
            # images are float32 0–255 from image_dataset_from_directory
            yield [tf.cast(images, tf.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
    print(f"Wrote quantized TFLite model: {TFLITE_PATH} ({size_mb:.2f} MB)")


def main():
    print(f"TensorFlow {tf.__version__}")
    class_names = require_dataset()
    print(f"Classes: {class_names}")

    train_ds, val_ds = load_datasets()
    model, base = build_model(num_classes=len(class_names))
    model.summary()

    model = train(model, base, train_ds, val_ds)

    print("\n=== Exporting labels and quantized TFLite model ===")
    export_labels(class_names)
    export_quantized_tflite(model, train_ds)
    print("Done.")


if __name__ == "__main__":
    main()
