from lib.load.load_data import load_data, load_embedding
from lib.load.load_model import load_model
from lib.process.process_data import process_data
from lib.train.train_model import train_model
from lib.predict.predict import predict
from lib.evaluate.model_performance import evaluate_multiclass_model
from lib.evaluate.search_performance import embed_text, tsne_embedding, top_k_matches

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

OUTPUT_DIR = '/app/output'

def main_workflow(
        train = False
):
    # load data and embedding
    data = load_data()
    logging.info('success: loaded data')
    embedding = load_embedding()
    logging.info('success: loaded embedding')


    # prepare data for training and prediction
    data = process_data(
        data=data,
        embedding=embedding,
        max_len=int(1e3)
    )
    logging.info('success: processed data')

    # train model and predict or only predict
    if train:
        model, training_history = train_model(
            data = data['train'],
            label_mapping=data['label_mapping']
        )

        predictions = predict(
            data=data['test'],
            model=model
        )
        logging.info('success: generated predictions from newly trained model instance')
    else:
        model, model_name = load_model()

        predictions = predict(
            data=data['test'],
            model=model
        )

        logging.info(f'success: generated predictions from {model_name}')

    # evaluate model performance
    evaluate_multiclass_model(
        y_true=data['test']['y'],
        y_score=predictions,
        label_mapping=data['label_mapping'],
        output_dir=OUTPUT_DIR
    )

    # evaluate search performance
    trained_embedding = embed_text(
        data=data['test'],
        model=model
    )

    tsne_embedding(
        trained_embedding=trained_embedding,
        label_mapping=data['label_mapping'],
        output_dir=OUTPUT_DIR
    )

    top_k_matches(
        trained_embedding=trained_embedding,
        k=3,
        output_dir=OUTPUT_DIR
    )




