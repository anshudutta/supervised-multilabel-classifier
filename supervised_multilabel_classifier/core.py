import numpy as np
from gensim import utils
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from scipy import spatial


def find_match(vec2Ids, match, take=5):
    retrieval = []
    for i in range(len(vec2Ids)):
        retrieval.append((1 - spatial.distance.cosine(match, vec2Ids[i][0]), vec2Ids[i][1]))

    retrieval.sort(reverse=True)
    return retrieval[:take]


class AweVectorizer(object):
    def __init__(self, model):
        self.model = model
        self.punctuations = ['(', ')', ';', ':', '[', ']', ',']

    def get_tokens(self, text):
        tokens = utils.tokenize(remove_stopwords(text))
        keywords = [word for word in tokens if not word in self.punctuations]
        return keywords

    def transform(self, texts):
        return [self.get_vector(text) for text in texts]

    def get_vector(self, text):
        try:
            doc = [word for word in self.get_tokens(text) if word in self.model.vocab]
            return np.mean(self.model[doc], axis=0)
        except Exception as e:
            return np.zeros(300,)



class MultiLabelVectorizer(object):
    def __init__(self):
        self.mlb = None

    def transform(self, y):
        categories = [item for sublist in y for item in sublist]
        classes = list(set(categories))
        self.mlb = MultiLabelBinarizer(classes=classes)
        return self.mlb.fit_transform(y)

    def inverse_transform(self, vector):
        return self.mlb.inverse_transform(vector)


class Predictor(object):
    def __init__(self):
        self.classifier = OneVsRestClassifier(LinearSVC(random_state=0))

    def fit(self, x_train, y_train):
        self.classifier.fit(x_train, y_train)

    def predict(self, x):
        return self.classifier.predict(x)
