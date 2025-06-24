import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
import numpy as np
from lib.load.load_data import DataSplit

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def predict(
        data: DataSplit,
        model: tf.keras.Model
) -> np.ndarray:
    """ Generates predictions from a tf model

    Arguments
    ---------
        data: DataSplit from process_data() to generate predictions for
        model: tf model to generate predictions from

    Retunrns
    --------
        np.ndarray of predictions
    """

    return model.predict(data['x_embed'])