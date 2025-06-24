import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.manifold import TSNE
from typing import Dict
from lib.load.load_data import DataSplit



def extract_trained_embedding(
        model: tf.keras.Model
) -> tf.keras.Model:
    """Extracts trained / refined embedding layer from trained model

    Arguments
    ---------
        model: trained tf model to extract embedding layer from

    Returns
    -------
        embedding layer as tf model
    """
    embedding_layer = tf.keras.Model(
        inputs=model.get_layer(index=0).input,
        outputs=model.get_layer(index=-2).output
    )

    return embedding_layer

def embed_text(
        data: DataSplit,
        model: tf.keras.Model
) -> Dict:
    """ Returns trained / refined embedding of text

    Arguments
    ---------
        data: DataSplit from process_data()
        model: trained tf model to extract embedding layer from

    Returns
    -------
        Dict of np.ndarray of embedded text and np.array of associated text lables
    """
    embedding_model = extract_trained_embedding(model)
    embedded_text = embedding_model.predict(data['x_embed'], batch_size=128)

    return {'embeddings': embedded_text, 'x': data['x'], 'y': data['y']}

def tsne_embedding(
        trained_embedding: Dict,
        label_mapping: Dict,
        output_dir: str = '/app/output'
) -> None:
    """ plots tsne visualization of trained / refined embedding

    Arguments
    ---------
        trained_embedding: embedded texts from embed_text()
        label_mapping: Dict mapping numeric text labels to string text labels

    Returns
    -------
        None
    """
    tsne = TSNE(n_components=2, metric='cosine')
    tsne = tsne.fit_transform(trained_embedding['embeddings'])

    labels = np.array([label_mapping[i] for i in trained_embedding['y']])
    n_classes = len(label_mapping)
    cmap = get_cmap('viridis', n_classes)

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        idx = labels == label_mapping[i]
        plt.scatter(tsne[idx, 0], tsne[idx, 1], label=f'{label_mapping[i]}', alpha=0.7, s=30, color=cmap(i))

    plt.title("t-SNE of Trained Embedding")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'tsne_embedding.png'))


def top_k_matches(
        trained_embedding: Dict,
        k: int = 3,
        output_dir: str = '/app/output'
) -> None:
    embedding = trained_embedding['embeddings']
    norm_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    similarity = np.dot(norm_embedding, norm_embedding.T)

    # exclude self-similarity, set diagonal to -inf
    np.fill_diagonal(similarity, -np.inf)

    # get top-k indices per row
    top_k_indices = np.argpartition(-similarity, kth=k, axis=1)[:, :k]

    # sort top-k indices per row by actual similarity
    row_indices = np.arange(embedding.shape[0])[:, None]
    sorted_top_k = top_k_indices[row_indices, np.argsort(-similarity[row_indices, top_k_indices])]

    text_matches = trained_embedding['x'][sorted_top_k]
    text_matches_dict = {key: list(values) for key, values in zip(trained_embedding['x'], text_matches)}

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "top_k_search_matches.json"), "w") as f:
        json.dump(text_matches_dict, f, indent=2)



