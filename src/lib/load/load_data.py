import polars as pl
import numpy as np
from pathlib import Path
from typing import TypedDict, Dict, Optional
from gensim.models import KeyedVectors

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

class DataSplit(TypedDict):
    x: np.ndarray
    x_embed: Optional[np.ndarray]
    y: np.ndarray

class Data(TypedDict):
    test: DataSplit
    train: DataSplit
    label_mapping: Dict[int,str]

def load_data(path: Path | str = None) -> Data:
    """ Loads BBC news test and train data from local_path directory
    and text and numeric target label numpy arrays as well as a dictionary
    mapping from numeric to text labels.

    Arguments
    ---------
        path: Path to the directory containing parquet files.
        If None, defaults to `src/data`.

    Returns
    -------
        Dictionary containing test and train dictionaries with text
        and label arrays, as well as a label mapping dictionary.
    """

    if path is None:
        data_dir = Path(__file__).resolve().parents[2] / 'data'
    else:
        data_dir = Path(path)


    test = pl.read_parquet(data_dir / 'bbc_news_test.parquet')
    train = pl.read_parquet(data_dir / 'bbc_news_train.parquet')
    logging.info(f'loaded data from {data_dir}')

    label_dict = (
        pl.concat([test,train])
        .select(['label', 'label_text'])
        .unique()
        .sort('label')
    )

    label_dict = dict(
        zip(label_dict['label'].to_list(),
            label_dict['label_text'].to_list())
    )

    x_test = test['text'].to_numpy().astype(np.str_)
    x_train = train['text'].to_numpy().astype(np.str_)

    y_test = test['label'].to_numpy().astype(np.int64)
    y_train = train['label'].to_numpy().astype(np.int64)

    # prepare output
    out = {
       'test': {
          'x': x_test,
          'x_embed': None,
          'y': y_test
       },
       'train': {
          'x': x_train,
          'x_embed': None,
          'y': y_train
       },
       'label_mapping': label_dict
    }

    logging.info(f'prepared data')

    return out


def load_embedding(
        path: Path | str = None,
        embedding_name: str = 'GoogleNews-vectors-negative300.bin.gz'
) -> KeyedVectors:
    """ Loads a pretrained embedding binary specified by embedding_name from
    a directory specified by path.

    Arguments
    ---------
        path: Path to the directory containing the embedding binary.
        If None defaults to 'src/data'
        embedding_name: Name of the embedding binary file.
        If None defaults to GoogleNews negative 300.

    Returns
    -------
        KeyedVectors
    """

    if path is None:
        embedding_dir = Path(__file__).resolve().parents[2] / 'data'
    else:
        embedding_dir = Path(path)

    logging.info(f'loading {embedding_name} from {embedding_dir}')
    embedding = KeyedVectors.load_word2vec_format(embedding_dir / embedding_name, binary=True)
    logging.info(f'loaded embedding {embedding_name}')

    return embedding





