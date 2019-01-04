import sys
import warnings

from nltk.corpus import reuters
from sklearn.metrics import classification_report

from supervised_multilabel_classifier.core import Predictor, AweVectorizer, MultiLabelVectorizer, find_match
from supervised_multilabel_classifier.service import get_vectors_from_csv
from supervised_multilabel_classifier.service import load_model, get_vectors_from_reuters
from supervised_multilabel_classifier.utility import config_logger, Spinner


def get_filename_from_arg():
    if len(sys.argv) > 1:
        return sys.argv[1]
    else:
        return None


def main():
    print('Running multi-label text classifier.')
    warnings.filterwarnings("ignore")
    logger = config_logger()
    spinner = Spinner()

    print('Loading pre-trained model...')
    spinner.start()
    model = load_model()
    spinner.stop()

    filename = get_filename_from_arg()

    x_vec = AweVectorizer(model)
    y_vec = MultiLabelVectorizer()

    vec2ids = None

    if filename is None:
        print('No training set provided, running default mode on reuters corpus.')
        print('Training model...')
        spinner.start()
        x_train, x_test, y_train, y_true = get_vectors_from_reuters(x_vec, y_vec)
        spinner.stop()

    else:
        print('Training model...')
        spinner.start()
        x_train, x_test, y_train, y_true, vec2ids = get_vectors_from_csv(filename,
                                                                         ["ID", "Text", "Category"],
                                                                         x_vec, y_vec, 0.1)
        spinner.stop()

    predictor = Predictor()
    predictor.fit(x_train, y_train)
    y_predicted = predictor.predict(x_test)

    print('Training stats:')
    print(classification_report(y_true, y_predicted, target_names=reuters.categories()))

    while True:
        print("Enter/Paste your content. Ctrl-D or Ctrl-Z ( windows ) to save it.")
        contents = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            contents.append(line)

        text = ''.join(contents)
        if len(text) > 0:
            x_test = x_vec.transform([text])
            y_predicted = predictor.predict(x_test)
            predicted = y_vec.inverse_transform(y_predicted)
            print("prediction: {0}".format([",".join(p) for p in predicted]))
            if vec2ids is not None:
                print("matches:".format(find_match(vec2ids, x_test)))
        else:
            print("Invalid entry")


if __name__ == "__main__":
    main()
