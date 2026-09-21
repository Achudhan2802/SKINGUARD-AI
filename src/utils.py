import os
import json
import logging
from PIL import Image
from src.config import LOGS_PATH, OUTPUTS_PATH, REPORTS_PATH, UPLOADS_PATH, MODEL_PATH

def setup_logging():
    os.makedirs(LOGS_PATH, exist_ok=True)
    log_file = os.path.join(LOGS_PATH, "app.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

def create_directories():
    for path in [OUTPUTS_PATH, REPORTS_PATH, UPLOADS_PATH, LOGS_PATH, os.path.dirname(MODEL_PATH)]:
        os.makedirs(path, exist_ok=True)

def save_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def validate_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False