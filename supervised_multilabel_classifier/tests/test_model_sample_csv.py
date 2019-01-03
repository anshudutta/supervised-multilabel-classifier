import os
import pytest
from supervised_multilabel_classifier import core
from supervised_multilabel_classifier import service
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from nltk.corpus import reuters


@pytest.fixture
def test_fixture():
    file_name = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_set/training_data.csv'))

    model = service.load_model()

    x_vec = core.AweVectorizer(model)
    y_vec = core.MultiLabelVectorizer()

    x, y, vec2ids = service.get_vectors_from_csv(file_name, ["ID", "Text", "Category"], x_vec, y_vec)
    pytest.x, pytest.y, pytest.vec2ids = x, y, vec2ids
    pytest.x_vec, pytest.y_vec = x_vec, y_vec


def test_model(test_fixture):
    x_train, x_test, y_train, y_test = train_test_split(pytest.x, pytest.y, test_size=0.1)

    predictor = core.Predictor()
    predictor.fit(x_train, y_train)
    y_predicted = predictor.predict(x_test)
    accuracy = accuracy_score(y_test, y_predicted)
    assert (accuracy > 0.5)


def test_prediction(test_fixture):
    y_test = [("ny",)]
    text = "New York is a lovely city although it is not a capital"
    assert_prediction(text, y_test)

    y_test = [("london", "paris")]
    text = "I am going to visit London and Paris"
    assert_prediction(text, y_test)


def assert_prediction(text, y_test):
    predictor = core.Predictor()
    predictor.fit(pytest.x, pytest.y)

    x_test = pytest.x_vec.transform([text])
    predicted = predictor.predict(x_test)
    y_predicted = pytest.y_vec.inverse_transform(predicted)
    matches = core.find_match(pytest.vec2ids, x_test)

    assert (y_predicted == y_test)
    assert (len(matches) > 0)
