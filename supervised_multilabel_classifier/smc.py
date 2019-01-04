import sys
import numpy as np
import warnings
from nltk.corpus import reuters
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from supervised_multilabel_classifier import reuters_nltk
from supervised_multilabel_classifier.service import load_model, get_vectors_from_reuters
from supervised_multilabel_classifier.core import Predictor, AweVectorizer, MultiLabelVectorizer, find_match
from supervised_multilabel_classifier.utility import config_logger, Spinner
from supervised_multilabel_classifier.service import get_vectors_from_csv


def get_filename_from_arg():
    if len(sys.argv) > 1:
        return sys.argv[1]
    else:
        return None


def main():
    warnings.filterwarnings("ignore")
    logger = config_logger()
    model = load_model()
    filename = get_filename_from_arg()
    x_vec = AweVectorizer(model)
    y_vec = MultiLabelVectorizer()
    vec2ids = None

    if filename is None:
        print('No training set location was provided, training with reuters corpus...')
        train_docs, test_docs, train_categories, test_categories = reuters_nltk.get_docs()

        spinner = Spinner()
        spinner.start()
        x_train, x_test, y_train, y_true = get_vectors_from_reuters(train_docs, train_categories,
                                                                    test_categories, x_vec, y_vec)
        spinner.stop()

    else:

        x, y, vec2ids = get_vectors_from_csv(filename, ["ID", "Text", "Category"], x_vec, y_vec)
        x_train, x_test, y_train, y_true = train_test_split(x, y, test_size=0.1)
        x_train, x_test, y_train, y_true = np.asarray(x_train), np.asarray(x_test), np.asarray(y_train), np.asarray(
            y_true)

    predictor = Predictor()
    predictor.fit(x_train, y_train)
    y_predicted = predictor.predict(x_test)

    print('Training stats:')
    print(classification_report(y_true, y_predicted, target_names=reuters.categories()))

    while True:
        text = ""
        print("Enter your text:")
        for line in sys.stdin:
            if line.strip() == '0':
                break
                text += line
        else:
            print('Warning: EOF occurred before "0" terminator')

        if len(text) > 0:
            x_test = x_vec.transform([text])
            y_predicted = predictor.predict(x_test)
            predicted = y_vec.inverse_transform(y_predicted)
            print("prediction: {0}".format([",".join(p) for p in predicted]))
            if vec2ids is not None:
                print("matches:".format(find_match(vec2ids, x_test)))
        cont = input("Do you want to continue? y/n:")
        if cont == "y":
            continue


if __name__ == "__main__":
    main()
