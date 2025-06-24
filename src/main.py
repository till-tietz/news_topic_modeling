import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import argparse

from workflows.main_workflow import main_workflow


def parse_args():
    parser = argparse.ArgumentParser(description='news topic modeling workflow')

    parser.add_argument(
        '--train',
        type=str,
        choices=['true','false'],
        required=True,
        help='Logical flag indicating whether to train a new model instance or use an old one'
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # handle output destinations based on environment
    if args.train == 'true':
        train = True
    elif args.train == 'false':
        train = False
    else:
        raise ValueError(f"Unsupported train argument: {args.train}")

    # run prediction workflow
    main_workflow(
        train=train
    )




if __name__ == "__main__":
    main()