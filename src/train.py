import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from src.config import MODEL_PATH, CLASS_NAMES_PATH, HISTORY_PATH, INITIAL_EPOCHS, FINE_TUNE_EPOCHS, INITIAL_LEARNING_RATE, FINE_TUNE_LEARNING_RATE
from src.data_preprocessing import prepare_dataset, create_tf_datasets
from src.model import build_efficientnet_model
from src.utils import setup_logging, create_directories, save_json

def add_dummy_metadata(images, labels):
    """
    Injects a zero-filled tensor of shape (batch_size, 4) to match 
    the model's meta_input branch during training.
    """
    batch_size = tf.shape(images)[0]
    dummy_meta = tf.zeros((batch_size, 4), dtype=tf.float32)
    # Return format must be a nested tuple: ((image_inputs, meta_inputs), labels)
    return (images, dummy_meta), labels

def train():
    setup_logging()
    create_directories()
    
    print("[INFO] Preparing dataset and splitting dataframe...")
    train_df, val_df, test_df, class_weights, idx_to_class = prepare_dataset()
    
    print("[INFO] Creating TensorFlow datasets (this caches and maps images)...")
    train_ds, val_ds, _ = create_tf_datasets(train_df, val_df, test_df)
    
    # Inject dummy metadata tensors into the training and validation datasets
    print("[INFO] Injecting dummy metadata pipelines for multi-input training...")
    train_ds = train_ds.map(add_dummy_metadata, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(add_dummy_metadata, num_parallel_calls=tf.data.AUTOTUNE)
    
    print(f"[INFO] Saving class names to {CLASS_NAMES_PATH}...")
    save_json(idx_to_class, CLASS_NAMES_PATH)
    
    print("[INFO] Building EfficientNet model...")
    model, base_model = build_efficientnet_model(num_classes=len(idx_to_class))
    
    # ==========================================
    # PHASE 1: Initial Training (Frozen Backbone)
    # ==========================================
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE), 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    weights_path = MODEL_PATH.replace('.keras', '.weights.h5')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint(weights_path, monitor='val_accuracy', save_best_only=True, save_weights_only=True, verbose=1)
    ]
    
    print("[INFO] Starting initial training phase (frozen base model)...")
    model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=INITIAL_EPOCHS, 
        class_weight=class_weights, 
        callbacks=callbacks,
        verbose=1
    )
    
    # ==========================================
    # PHASE 2: Fine-Tuning (Unfreezing Top Layers)
    # ==========================================
    print("[INFO] Unfreezing base model for high-accuracy fine-tuning...")
    base_model.trainable = True
    
    # Keep lower layers frozen, unfreeze the top 30 layers for skin lesion feature extraction
    for layer in base_model.layers[:-30]:
        layer.trainable = False
        
    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE), # Low LR (e.g., 1e-5) for stable fine-tuning
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    print("[INFO] Starting fine-tuning phase to hit 80% accuracy target...")
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=FINE_TUNE_EPOCHS, 
        class_weight=class_weights, 
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final optimized model after training completes
    model.save(MODEL_PATH)
    print(f"[SUCCESS] Full optimized model saved successfully to {MODEL_PATH}!")

if __name__ == '__main__':
    train()