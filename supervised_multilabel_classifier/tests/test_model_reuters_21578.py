import pytest
from supervised_multilabel_classifier import core
from supervised_multilabel_classifier.service import get_vectors_from_reuters, load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from nltk.corpus import reuters


# NOTE: Install reuters corpus using python console - nltk.download('reuters')

@pytest.fixture
def test_fixture():
    model = load_model()

    x_vec = core.AweVectorizer(model)
    y_vec = core.MultiLabelVectorizer()

    x_train, x_test, y_train, y_true, vec2ids = get_vectors_from_reuters(x_vec, y_vec)

    pytest.x_train, pytest.x_test, pytest.y_train, pytest.y_true = x_train, x_test, y_train, y_true
    pytest.x_vec, pytest.y_vec = x_vec, y_vec


def test_model(test_fixture):
    predictor = core.Predictor()
    predictor.fit(pytest.x_train, pytest.y_train)
    predicted = predictor.predict(pytest.x_test)
    accuracy = accuracy_score(pytest.y_true, predicted)
    # tp / (tp + fp) The precision is intuitively the ability of the classifier not to label as positive a sample
    # that is negative.
    precision = precision_score(pytest.y_true, predicted, average='macro')

    # tp / (tp + fn) The recall is intuitively the ability of the classifier to find all the positive samples.
    recall = recall_score(pytest.y_true, predicted, average='macro')
    print(classification_report(pytest.y_true, predicted, target_names=reuters.categories()))
    assert (accuracy > 0.7)

