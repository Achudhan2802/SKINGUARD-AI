import pandas as pd
import tensorflow as tf
from src.config import MODEL_PATH, CLASS_NAMES_PATH
from src.data_preprocessing import prepare_dataset, create_tf_datasets
from src.utils import load_json

def evaluate():
    print("[INFO] Preparing dataset for evaluation...")
    # Get the test dataframe
    train_df, val_df, test_df, class_weights, idx_to_class = prepare_dataset()
    
    print("[INFO] Creating test dataset pipeline...")
    # Pass test_df for all three to avoid the KeyError, we only keep test_ds
    _, _, test_ds = create_tf_datasets(test_df, test_df, test_df)
    
    print(f"[INFO] Loading trained model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print("[INFO] Evaluating model on test data...")
    loss, accuracy = model.evaluate(test_ds)
    print(f"\n[RESULT] Test Loss: {loss:.4f}")
    print(f"[RESULT] Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    evaluate()