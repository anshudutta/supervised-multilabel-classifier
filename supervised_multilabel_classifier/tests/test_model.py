import os
import pytest
from supervised_resume_matcher import utility
from supervised_resume_matcher import core
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@pytest.fixture
def test_fixture():
    pytest.file_name = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_set/training_data.csv'))

    logger = utility.config_logger()
    model = utility.load_model(50000)

    vec = core.Vectorizer(logger)
    pytest.predictor = core.Predictor(logger, vec, model)

    x, y = vec.get_vectors(pytest.file_name, "Text", "Category", model)
    pytest.x_train, pytest.x_test, pytest.y_train, pytest.y_test = train_test_split(x, y, test_size=0.2)
    pytest.predictor.fit(pytest.x_train, pytest.y_train)
    pytest.vec = vec
    pytest.model = model


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
    x_test = pytest.vec.get_average_word_embedding(text, pytest.model)
    y_predicted = pytest.vec.get_labels(pytest.predictor.predict([x_test]))
    assert (y_predicted == y_test)
