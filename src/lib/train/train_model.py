import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
from lib.load.load_data import DataSplit
from pathlib import Path
from datetime import datetime
from typing import Dict

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

def save_model(
        model: tf.keras.Model,
        path: Path | str = None,
) -> None:

    if path is None:
        model_dir = Path(__file__).resolve().parents[2] / 'models'
    else:
        model_dir = Path(path)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    model_filename = f'model_{timestamp}.keras'

    model.save(model_dir / model_filename)
    logging.info(f'saved {model_filename} to {model_dir}')


def train_model(
        data: DataSplit,
        label_mapping: Dict,
        epochs: int = 15,
        batch_size: int = 128,
        model_path: Path | str = None
) -> tf.keras.Model:
    """ Trains a tf LSTM classifier

    Arguments:
        data: a DataSplit from the output of process_data()
        label_mapping: mapping of outcome labels from process_data()
        epochs: int number of training epochs
        batch_size: int batch size
        model_path: path to save model to. If None defaults to 'src/models'

    Returns:
        A trained tf model and its training history
    """
    _, max_len, embded_dim = data['x_embed'].shape
    n_categories = len(label_mapping)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Masking(mask_value=0., input_shape=(max_len, embded_dim)))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    model.add(tf.keras.layers.LSTM(128, return_sequences=False))
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Dense(n_categories, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        data['x_embed'],
        data['y'],
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )

    save_model(model, model_path)

    return model, history.history