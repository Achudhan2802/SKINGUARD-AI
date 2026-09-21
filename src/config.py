import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Your exact path where all images are stored directly
DATASET_PATH = r"C:\Users\achud\Documents\Skin Cancer Project\dataset\HAM10000_images"

# Metadata path (ensure HAM10000_metadata.csv is a real file, not a folder, inside the dataset folder)
METADATA_PATH = r"C:\Users\achud\Documents\Skin Cancer Project\dataset\HAM10000_metadata.csv"

# Model and output paths relative to your base directory
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "models", "class_names.json")
HISTORY_PATH = os.path.join(BASE_DIR, "models", "training_history.json")

OUTPUTS_PATH = os.path.join(BASE_DIR, "outputs")
REPORTS_PATH = os.path.join(BASE_DIR, "reports")
UPLOADS_PATH = os.path.join(BASE_DIR, "uploads")
LOGS_PATH = os.path.join(BASE_DIR, "logs")

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 7
BATCH_SIZE = 32
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 10
INITIAL_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
RANDOM_SEED = 42
CONFIDENCE_THRESHOLD = 0.70

LESION_TYPE_DICT = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses / intraepithelial carcinoma',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}