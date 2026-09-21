import os
import tensorflow as tf

# Updated to point to your actual saved model directory
MODEL_PATH = "models/best_model.keras"
TFLITE_PATH = "outputs/skinguard_model.tflite"


def convert_model():
  print(f"[INFO] Loading trained model from {MODEL_PATH}...")
  if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Train the model first!"
    )

  model = tf.keras.models.load_model(MODEL_PATH)

  print(
      "[INFO] Defining concrete input signatures for multi-input TFLite"
      " export..."
  )

  # Define exact expected shapes for mobile inference:
  # 1. Image input: Batch of 1, 224x224 pixels, 3 channels
  # 2. Metadata input: Batch of 1, 4 clinical features (Age, Sex, Duration, Scaliness)
  @tf.function(
      input_signature=[
          tf.TensorSpec(
              shape=(1, 224, 224, 3), dtype=tf.float32, name="image_input"
          ),
          tf.TensorSpec(shape=(1, 4), dtype=tf.float32, name="meta_input"),
      ]
  )
  def serve(image_input, meta_input):
    return model([image_input, meta_input])

  print("[INFO] Converting model to TensorFlow Lite format...")
  converter = tf.lite.TFLiteConverter.from_concrete_functions(
      [serve.get_concrete_function()]
  )

  # Optional: Enable default optimizations (quantization for smaller mobile file size)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]

  tflite_model = converter.convert()

  os.makedirs(os.path.dirname(TFLITE_PATH), exist_ok=True)
  with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

  print(f"[SUCCESS] TFLite model successfully saved to {TFLITE_PATH}!")


if __name__ == "__main__":
  convert_model()