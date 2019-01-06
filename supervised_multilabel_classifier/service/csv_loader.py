from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from supervised_multilabel_classifier.service.data_loader import DataLoader


class CsvLoader(DataLoader):

    def __init__(self, filename, cols):
        self.filename, self.cols = filename, cols,

    def get_vectors(self,  x_vec, y_vec, test_size=0.1):
        texts, categories, ids = read(self.filename, [self.cols[0]], [self.cols[1]], [self.cols[2]])
        y_train = y_vec.transform(categories)
        x_train = x_vec.transform(texts)

        x_train, x_test, y_train, y_true = train_test_split(x_train, y_train, test_size=test_size)
        return np.asarray(x_train), np.asarray(x_test), np.asarray(y_train), np.asarray(y_true), super().get_vec_to_id(x_train, ids)


def read(source, id_col, x_col, y_col):
    reader = pd.read_csv(source)
    return process_cols(reader[x_col], reader[y_col], reader[id_col])


def process_cols(x, y, z):
    return process_x(x), process_y(y), z.values.flatten()


def process_x(x):
    return [text[0] for text in x.values]


def process_y(y):
    return [[c.strip() for c in classes[0].split(",")] for classes in y.values]
