import os
import gensim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from supervised_multilabel_classifier.reuters_nltk import get_docs


def get_vectors_from_csv(file_name, cols, x_vec, y_vec, test_size):
    texts, categories, ids = read(file_name, [cols[0]], [cols[1]], [cols[2]])
    y_train = y_vec.transform(categories)
    x_train = x_vec.transform(texts)

    x_train, x_test, y_train, y_true = train_test_split(x_train, y_train, test_size=test_size)
    return np.asarray(x_train), np.asarray(x_test), np.asarray(y_train), np.asarray(y_true), get_vec_to_id(x_train, ids)


def get_vectors_from_reuters(x_vec, y_vec):
    train_docs, test_docs, train_categories, test_categories, ids = get_docs()
    x_train = x_vec.transform(train_docs)
    x_test = x_vec.transform(test_docs)
    y_train = y_vec.transform(train_categories)
    y_true = y_vec.transform(test_categories)

    return x_train, x_test, y_train, y_true, get_vec_to_id(x_train, ids)


def get_vec_to_id(x_train, ids):
    vec2id = []
    for idx, x in enumerate(x_train):
        vec2id.append((x, ids[idx]))
    return vec2id


def load_model(limit=None):
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'model'))
    raw_model_path = os.path.join(model_path, 'GoogleNews-vectors-negative300.bin')
    norm_model_path = os.path.join(model_path, 'GoogleNews-vectors-gensim-normed.bin')

    if not os.path.isfile(norm_model_path):
        if limit is None:
            raw_model = gensim.models.KeyedVectors.load_word2vec_format(raw_model_path, binary=True)
        else:
            raw_model = gensim.models.KeyedVectors.load_word2vec_format(raw_model_path, binary=True, limit=limit)
        raw_model.save(norm_model_path)
    model = gensim.models.KeyedVectors.load(norm_model_path, mmap='r')
    model.vectors_norm = model.vectors
    return model


def read(source, id_col, x_col, y_col):
    reader = pd.read_csv(source)
    return process_cols(reader[x_col], reader[y_col], reader[id_col])


def process_cols(x, y, z):
    return process_x(x), process_y(y), z.values.flatten()


def process_x(x):
    return [text[0] for text in x.values]


def process_y(y):
    return [[c.strip() for c in classes[0].split(",")] for classes in y.values]
