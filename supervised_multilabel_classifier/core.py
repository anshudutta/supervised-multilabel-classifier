import numpy as np
from gensim import utils
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from scipy import spatial
from supervised_resume_matcher import utility


class Vectorizer(object):
    def __init__(self, logger):
        self.logger = logger
        self.punctuations = ['(', ')', ';', ':', '[', ']', ',']
        self.mlb = None

    def get_one_hot_encoding(self, y):
        categories = [item for sublist in y for item in sublist]
        classes = list(set(categories))
        self.mlb = MultiLabelBinarizer(classes=classes)
        return self.mlb.fit_transform(y)

    def get_labels(self, prediction):
        return self.mlb.inverse_transform(prediction)

    def get_tokens(self, text):
        tokens = utils.tokenize(remove_stopwords(text))
        keywords = [word for word in tokens if not word in self.punctuations]
        return keywords

    def get_average_word_embedding(self, text, model):
        doc = [word for word in self.get_tokens(text) if word in model.wv.vocab]
        return np.mean(model.wv[doc], axis=0)

    def get_vectors(self, file_name, x_col, y_col, model):
        texts, categories = utility.read(file_name, [x_col], [y_col])
        y_train = self.get_one_hot_encoding(categories)
        x_train = [self.get_average_word_embedding(text[0], model) for text in texts]
        return x_train, y_train


class Predictor(object):
    def __init__(self, log, vectorizer, model):
        self.log = log
        self.vectorizer = vectorizer
        self.classifier = None
        self.model = model

    def fit(self, x_train, y_train):
        self.classifier = OneVsRestClassifier(LinearSVC(random_state=0))
        self.classifier.fit(x_train, y_train)

    def predict(self, x):
        return self.classifier.predict(x)

    def find_match(self, vectors, vector):
        retrieval = []

        for i in range(len(vectors)):
            retrieval.append((1 - spatial.distance.cosine(vector, vectors[i][0]), vectors[i][1]))

        retrieval.sort(reverse=True)
        return retrieval
