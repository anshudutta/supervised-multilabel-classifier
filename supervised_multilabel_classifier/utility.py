import os
import sys
import logging
import gensim
import pandas as pd


def get_file_name(relative_path):
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    return os.path.join(file_dir, relative_path)


def config_logger():
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', filename='algo.log', level=logging.INFO)
    return logger


def load_model(limit=300000):
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'model'))
    raw_model_path = os.path.join(model_path, 'GoogleNews-vectors-negative300.bin')
    norm_model_path = os.path.join(model_path, 'GoogleNews-vectors-gensim-normed.bin')

    if not os.path.isfile(norm_model_path):
        raw_model = gensim.models.KeyedVectors.load_word2vec_format(raw_model_path, binary=True, limit=limit)
        raw_model.save(norm_model_path)
    model = gensim.models.KeyedVectors.load(norm_model_path, mmap='r')
    model.syn0norm = model.syn0
    return model


def read(source, x_col, y_col):
    reader = pd.read_csv(source)
    return process_x_y(reader[x_col], reader[y_col])


def process_x_y(x, y):
    return x.values, [[c.strip() for c in classes[0].split(",")] for classes in y.values]
