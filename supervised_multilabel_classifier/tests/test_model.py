import os
import pytest
from supervised_multilabel_classifier import utility
from supervised_multilabel_classifier import core
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@pytest.fixture
def test_fixture():
    pytest.file_name = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_set/training_data.csv'))

    logger = utility.config_logger()
    model = utility.load_model(50000)

    vec = core.Vectorizer(logger, model)
    x, y, pytest.vec2ids = vec.get_vectors(pytest.file_name, "ID", "Text", "Category")
    pytest.x_train, pytest.x_test, pytest.y_train, pytest.y_test = train_test_split(x, y, test_size=0.2)

    pytest.predictor = core.Predictor(logger)
    pytest.predictor.fit(pytest.x_train, pytest.y_train)

    pytest.vec = vec


def test_model(test_fixture):
    y_predicted = pytest.predictor.predict(pytest.x_test)
    accuracy = accuracy_score(pytest.y_test, y_predicted)
    assert (accuracy > 0.8)


def test_prediction(test_fixture):
    y_test = [("ny",)]
    text = "New York is a lovely city although it is not a capital"
    assert_prediction(text, y_test)

    y_test = [("london", "paris")]
    text = "I am going to visit London and Paris"
    assert_prediction(text, y_test)


def assert_prediction(text, y_test):
    vec = pytest.vec
    x_test = vec.get_average_word_embedding(text)
    y_predicted = vec.get_classes_from_vector(pytest.predictor.predict([x_test]))
    matches = pytest.predictor.find_match(pytest.vec2ids, x_test)

    assert (y_predicted == y_test)
    assert (len(matches) > 0)
