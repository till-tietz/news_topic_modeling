import os
import re
from datetime import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
from pathlib import Path
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

def load_model(
        path: Path | str = None,
        model_name: str = None
) -> tf.keras.Model:
    """ Loads tf model specified by model_name from a directory specified by
    model path. If both arguments are None it loads the latest model from the
    default directory 'src/models'.

    Arguments
    ---------
        path: Path to the directory containing the saved model.
        If None, defaults to 'src/models'.
        model_name: Name of the model to be loaded.
        If None defaults to model with latest timestamp.

    Returns
    -------
        tf.keras.Model
    """

    if path is None:
        model_dir = Path(__file__).resolve().parents[2] / 'models'
    else:
        model_dir = Path(path)


    if model_name is None:
        pattern = re.compile(r"([^_/]+)_(\d{8}T\d{6})\.keras$")

        latest_time = None
        latest_model_file = None

        for file in model_dir.iterdir():
            if file.is_file():
                match = pattern.search(file.name)
                if match:
                    timestamp_str = match.group(2)
                    try:
                        model_time = datetime.strptime(timestamp_str, "%Y%m%dT%H%M%S")
                        if latest_time is None or model_time > latest_time:
                            latest_time = model_time
                            latest_model_file = file
                    except ValueError:
                        continue

        if latest_model_file:
            model_name = latest_model_file
            path = latest_model_file
            logging.info(f"Loading latest model trained on: {latest_time} from {path}")
        else:
            raise FileNotFoundError("No valid model file found in local directory.")
    else:
        path = model_dir / model_name
        logging.info(f"Loading model from: {path}")


    model = tf.keras.models.load_model(path)
    logging.info('loaded model')

    return model, model_name

