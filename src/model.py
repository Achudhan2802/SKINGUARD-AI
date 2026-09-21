import tensorflow as tf
from tensorflow.keras import layers, Model
from src.config import NUM_CLASSES, IMAGE_SIZE


def build_efficientnet_model(num_classes=NUM_CLASSES, dropout_rate=0.4):
    """
    Builds an EfficientNetB0-based multimodal classifier that accepts 
    dermoscopic images and 4 clinical metadata features (Age, Sex, Duration, Scaliness)
    with fine-tuning enabled for higher accuracy performance (~80%+ target).
    """
    input_shape = (*IMAGE_SIZE, 3)

    # 1. Image Branch (EfficientNetB0 Backbone)
    base_model = tf.keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
        pooling=None,
    )
    
    # CRITICAL UPDATE: Unfreeze the backbone for fine-tuning
    base_model.trainable = True
    
    # Freeze the initial layers, but keep the top 30 layers trainable 
    # so the network can learn specific dermatological structures (pigment networks, streaks)
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    image_inputs = layers.Input(shape=input_shape, name="image_input")
    # Set training=True to ensure BatchNormalization layers adapt during fine-tuning
    x = base_model(image_inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    # 2. Metadata / Patient Record Branch (4 features: Age, Sex, Duration, Scaliness)
    meta_input_dim = 4
    meta_inputs = layers.Input(shape=(meta_input_dim,), name="meta_input")
    y = layers.Dense(32, activation='relu')(meta_inputs)
    y = layers.Dense(16, activation='relu')(y)

    # 3. Feature Fusion & Classification
    combined = layers.Concatenate()([x, y])
    z = layers.Dense(64, activation='relu')(combined)
    z = layers.Dropout(dropout_rate)(z)
    outputs = layers.Dense(num_classes, activation='softmax', name="classification_output")(z)

    # 4. Multi-Input Model Definition
    model = Model(inputs=[image_inputs, meta_inputs], outputs=outputs, name="SkinGuard_Multimodal_Model")
    
    return model, base_model