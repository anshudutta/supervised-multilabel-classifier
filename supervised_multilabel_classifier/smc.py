import argparse
import warnings

from sklearn.metrics import classification_report

from supervised_multilabel_classifier.core import Predictor, AweVectorizer, MultiLabelVectorizer, find_match
from supervised_multilabel_classifier.service.csv_loader import CsvLoader
from supervised_multilabel_classifier.service.model_loader import load_model
from supervised_multilabel_classifier.service.reuters_loader import ReutersLoader
from supervised_multilabel_classifier.utility import config_logger, Spinner


def get_filename_from_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=argparse.FileType('r', encoding='UTF-8'), required=False)
    args = parser.parse_args()
    return args.model


def get_multi_line_input():
    print("Enter/Paste your content. Ctrl-D or Ctrl-Z ( windows ) to save it.")
    contents = []
    while True:
        try:
            line = input()
            contents.append(line)
        except EOFError:
            break

    return ''.join(contents)


def main():
    print('Running multi-label text classifier.')
    warnings.filterwarnings("ignore")

    filename = get_filename_from_arg()

    logger = config_logger()
    spinner = Spinner()

    print('Loading pre-trained model...')
    spinner.start()
    model = load_model()
    spinner.stop()

    x_vec = AweVectorizer(model)
    y_vec = MultiLabelVectorizer()

    if filename is None:
        print('No training set provided, switching to default mode - reuters corpus.')
        data_loader = ReutersLoader()
    else:
        data_loader = CsvLoader(filename, ["ID", "Text", "Category"])

    print('Loading training data model...')
    spinner.start()
    x_train, x_test, y_train, y_true, vec2ids = data_loader.get_vectors(x_vec, y_vec)
    spinner.stop()

    print('Training model...')
    predictor = Predictor()
    predictor.fit(x_train, y_train)
    y_predicted = predictor.predict(x_test)
    print('Finished training model. Stats: ')
    print(classification_report(y_true, y_predicted, target_names=y_vec.get_classes()))

    text = get_multi_line_input()

    if len(text) > 0:
        x_test = x_vec.transform([text])
        y_predicted = predictor.predict(x_test)
        predicted = y_vec.inverse_transform(y_predicted)
        print("prediction: {0}".format(predicted))
        matches = find_match(vec2ids, x_test)
        print("Most similar: {0}".format(matches))
    else:
        print("Invalid entry")


if __name__ == "__main__":
    main()
