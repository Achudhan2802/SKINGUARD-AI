import numpy as np
import tensorflow as tf
from PIL import Image
from src.config import MODEL_PATH, CLASS_NAMES_PATH, IMAGE_SIZE, CONFIDENCE_THRESHOLD, LESION_TYPE_DICT
from src.utils import load_json

class Predictor:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.idx_to_class = {int(k): v for k, v in load_json(CLASS_NAMES_PATH).items()}
        
    def predict_image(self, image_path):
        img = Image.open(image_path).convert('RGB').resize(IMAGE_SIZE)
        img_array = np.expand_dims(tf.keras.applications.efficientnet.preprocess_input(np.array(img)), axis=0)
        preds = self.model.predict(img_array)[0]
        pred_idx = int(np.argmax(preds))
        confidence = float(preds[pred_idx])
        
        class_code = self.idx_to_class[pred_idx]
        return {
            "predicted_class": LESION_TYPE_DICT.get(class_code, class_code),
            "confidence": confidence,
            "probabilities": {LESION_TYPE_DICT.get(self.idx_to_class[i], self.idx_to_class[i]): float(preds[i]) for i in range(len(preds))},
            "uncertain": confidence < CONFIDENCE_THRESHOLD
        }