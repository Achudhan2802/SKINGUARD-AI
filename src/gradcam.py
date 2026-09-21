import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2

def generate_gradcam_images(image_path, model):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    original_img = tf.keras.utils.img_to_array(img)
    array = np.expand_dims(original_img, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(array)
    
    last_conv_layer_name = "top_conv"
    grad_model = Model(inputs=model.inputs, outputs=[model.get_layer(last_conv_layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, tf.argmax(predictions[0])]
        
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    heatmap = cv2.resize(np.uint8(255 * heatmap.numpy()), (224, 224))
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(original_img.astype(np.uint8), 0.6, heatmap_colored, 0.4, 0)
    return original_img, heatmap_colored, superimposed