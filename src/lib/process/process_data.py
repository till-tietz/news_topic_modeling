from lib.load.load_data import DataSplit, Data
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import KeyedVectors
import numpy as np

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

class embedding_transformer(BaseEstimator, TransformerMixin):
    def __init__(self, model, max_len=100):
        self.model = model
        self.dim = model.vector_size
        self.max_len = max_len

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = np.zeros((len(X), self.max_len, self.dim))

        for i, doc in enumerate(X):
            words = doc.lower().split()
            word_vectors = []
            for word in words:
                if word in self.model:
                    word_vectors.append(self.model[word])
                if len(word_vectors) == self.max_len:
                    break

            for j, vec in enumerate(word_vectors):
                result[i, j, :] = vec

        return result


def process_data(
        data: Data,
        embedding: KeyedVectors,
        max_len: int
) -> Data:
    """ Adds text embeddings to data

    Arguments
    ---------
        data: output of load_data()
        embedding: KeyedVectors loaded with load_embedding()
        max_len: maximum text lenght to truncate or pad embedded texts to

    Returns
    -------
        data with embeddings
    """

    transfomer = embedding_transformer(embedding, max_len=max_len)

    logging.info(f'generating embeddings')
    test_x_embed = transfomer.transform(data['test']['x'])
    train_x_embed = transfomer.transform(data['train']['x'])

    data['test']['x_embed'] = test_x_embed
    data['train']['x_embed'] = train_x_embed
    logging.info('added embeddings to data')

    return data
