import os
import gensim
import pandas as pd


def get_vectors_from_csv(file_name, cols, x_vec, y_vec):
    texts, categories, ids = read(file_name, [cols[0]], [cols[1]], [cols[2]])
    y_train = y_vec.transform(categories)
    x_train = x_vec.transform(texts)
    vec2id = []
    for idx, x in enumerate(x_train):
        vec2id.append((x, ids[idx][0]))
    return x_train, y_train, vec2id


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


def read(source, id_col, x_col, y_col):
    reader = pd.read_csv(source)
    return process_cols(reader[x_col], reader[y_col], reader[id_col])


def process_cols(x, y, z):
    return x.values, [[c.strip()[0] for c in classes[0].split(",")] for classes in y.values], z.values
