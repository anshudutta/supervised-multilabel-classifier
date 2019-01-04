import os

import pytest
from sklearn.metrics import accuracy_score

from supervised_multilabel_classifier import core
from supervised_multilabel_classifier import service


@pytest.fixture
def test_fixture():
    filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_set/training_data.csv'))
    model = service.load_model()

    x_vec = core.AweVectorizer(model)
    y_vec = core.MultiLabelVectorizer()

    pytest.x_vec, pytest.y_vec = x_vec, y_vec
    pytest.filename = filename


def test_model(test_fixture):
    x_train, x_test, y_train, y_true, vec2ids = service.get_vectors_from_csv(pytest.filename,
                                                                             ["ID", "Text", "Category"],
                                                                             pytest.x_vec, pytest.y_vec, 0.1)

    predictor = core.Predictor()
    predictor.fit(x_train, y_train)
    y_predicted = predictor.predict(x_test)
    accuracy = accuracy_score(y_true, y_predicted)
    assert (accuracy > 0.5)


def test_prediction(test_fixture):
    y_test = [("ny",)]
    text = "New York is a lovely city although it is not a capital"
    assert_prediction(text, y_test)

    y_test = [("london", "paris")]
    text = "I am going to visit London and Paris"
    assert_prediction(text, y_test)


def assert_prediction(text, y_true):
    x_train, x_d, y_train, y_d, vec2ids = service.get_vectors_from_csv(pytest.filename,
                                                                       ["ID", "Text", "Category"],
                                                                       pytest.x_vec, pytest.y_vec, 0)
    predictor = core.Predictor()
    predictor.fit(x_train, y_train)

    x_test = pytest.x_vec.transform([text])
    predicted = predictor.predict(x_test)
    y_predicted = pytest.y_vec.inverse_transform(predicted)
    matches = core.find_match(vec2ids, x_test)

    assert (check_equal(y_predicted, y_true) is True)
    assert (len(matches) > 0)


def check_equal(l1, l2):
    return len(l1) == len(l2) and sorted(l1) == sorted(l2)
