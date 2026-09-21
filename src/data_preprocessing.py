import os
import pandas as pd
import numpy as np
import tensorflow as tf
import logging
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import METADATA_PATH, DATASET_PATH, IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED, OUTPUTS_PATH
from src.utils import setup_logging

logger = logging.getLogger(__name__)

VALID_EXTS = ('.jpg', '.jpeg', '.png')


def load_metadata():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")
    return pd.read_csv(METADATA_PATH)


def build_image_index(dataset_dir):
    """
    Walk the dataset directory ONCE and build a dict of
    image_id -> full path. This replaces the old approach of calling
    os.listdir() + os.path.exists() for every single row, which turns
    an O(n_files) job into an O(n_images * n_files) job and is the
    main reason preprocessing was taking so long.
    """
    index = {}
    for root, _dirs, files in os.walk(dataset_dir):
        for fname in files:
            name, ext = os.path.splitext(fname)
            if ext.lower() in VALID_EXTS:
                # first match wins; adjust if you need to prefer a specific subfolder
                index.setdefault(name, os.path.join(root, fname))
    return index


def prepare_dataset():
    setup_logging()
    df = load_metadata()
    lesion_classes = sorted(df['dx'].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(lesion_classes)}
    idx_to_class = {idx: cls for idx, cls in enumerate(lesion_classes)}
    df['target'] = df['dx'].map(class_to_idx)

    # Build the path index once, then do a vectorized map instead of iterrows()
    image_index = build_image_index(DATASET_PATH)
    df['path'] = df['image_id'].map(image_index)

    n_missing = df['path'].isna().sum()
    if n_missing:
        logger.warning(f"{n_missing} images referenced in metadata were not found on disk")
    df = df.dropna(subset=['path']).reset_index(drop=True)

    os.makedirs(OUTPUTS_PATH, exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='dx', order=sorted(lesion_classes), palette='viridis')
    plt.title("HAM10000 Class Distribution")
    plt.savefig(os.path.join(OUTPUTS_PATH, "class_distribution.png"))
    plt.close()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, temp_idx = next(splitter.split(df, groups=df['lesion_id']))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    splitter_val = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_SEED)
    val_idx, test_idx = next(splitter_val.split(temp_df, groups=temp_df['lesion_id']))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df['target']),
        y=train_df['target'].values
    )
    return train_df, val_df, test_df, dict(enumerate(class_weights)), idx_to_class


def load_and_preprocess_image(path, label, training=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    if training:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img, label


def create_tf_datasets(train_df, val_df, test_df, cache_dir=None):
    """
    cache_dir: if provided, caches decoded/preprocessed images to disk
    (e.g. os.path.join(OUTPUTS_PATH, 'tfcache')) instead of memory, which
    is safer for datasets that don't fit in RAM. Pass None to cache in
    memory (fine for HAM10000-sized datasets at typical image sizes).
    """
    def cache_path(name):
        if cache_dir is None:
            return ''  # in-memory cache
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, name)

    train_ds = tf.data.Dataset.from_tensor_slices((train_df['path'].values, train_df['target'].values))
    train_ds = train_ds.map(
        lambda p, l: load_and_preprocess_image(p, l, training=True),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    # Cache AFTER decode/resize (the expensive I/O + CPU work) so every
    # epoch after the first reuses cached tensors instead of hitting disk
    # and re-decoding JPEGs again.
    train_ds = train_ds.cache(cache_path('train'))
    train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_df['path'].values, val_df['target'].values))
    val_ds = val_ds.map(
        lambda p, l: load_and_preprocess_image(p, l, training=False),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    val_ds = val_ds.cache(cache_path('val'))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_df['path'].values, test_df['target'].values))
    test_ds = test_ds.map(
        lambda p, l: load_and_preprocess_image(p, l, training=False),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, test_ds