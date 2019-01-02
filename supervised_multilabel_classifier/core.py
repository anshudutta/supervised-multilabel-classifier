import numpy as np
from gensim import utils
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from scipy import spatial
from supervised_multilabel_classifier import utility


class Vectorizer(object):
    def __init__(self, logger, model):
        self.logger = logger
        self.model = model
        self.punctuations = ['(', ')', ';', ':', '[', ']', ',']
        self.mlb = None

    def get_one_hot_encoding(self, y):
        categories = [item for sublist in y for item in sublist]
        classes = list(set(categories))
        self.mlb = MultiLabelBinarizer(classes=classes)
        return self.mlb.fit_transform(y)

    def get_classes_from_vector(self, prediction):
        return self.mlb.inverse_transform(prediction)

    def get_tokens(self, text):
        tokens = utils.tokenize(remove_stopwords(text))
        keywords = [word for word in tokens if not word in self.punctuations]
        return keywords

    def get_average_word_embedding(self, text):
        doc = [word for word in self.get_tokens(text) if word in self.model.wv.vocab]
        return np.mean(self.model.wv[doc], axis=0)

    def get_vectors(self, file_name, id_col, x_col, y_col):
        texts, categories, ids = utility.read(file_name, [id_col], [x_col], [y_col])
        y_train = self.get_one_hot_encoding(categories)
        x_train = [self.get_average_word_embedding(text[0]) for text in texts]
        vec2id = [] #dict(zip(list(x_train), ids))
        return x_train, y_train, vec2id


class Predictor(object):
    def __init__(self, log):
        self.log = log
        self.classifier = OneVsRestClassifier(LinearSVC(random_state=0))

    def fit(self, x_train, y_train):
        self.classifier.fit(x_train, y_train)

    def predict(self, x):
        return self.classifier.predict(x)

    def find_match(self, vectors, match, vec2Ids):
        retrieval = []

        for i in range(len(vectors)):
            retrieval.append((1 - spatial.distance.cosine(match, vectors[i][0]), vectors[i][1]))

        retrieval.sort(reverse=True)
        return retrieval
