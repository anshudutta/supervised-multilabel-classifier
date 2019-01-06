from nltk.corpus import reuters
from supervised_multilabel_classifier.service.data_loader import DataLoader


class ReutersLoader(DataLoader):

    def get_vectors(self, x_vec, y_vec):
        train_docs, test_docs, train_categories, test_categories, ids = get_docs()
        x_train = x_vec.transform(train_docs)
        x_test = x_vec.transform(test_docs)
        y_train = y_vec.transform(train_categories)
        y_true = y_vec.transform(test_categories)

        return x_train, x_test, y_train, y_true, super().get_vec_to_id(x_train, ids)


def get_docs():
    documents = reuters.fileids()

    train_docs_id = get_ids("train", documents)
    test_docs_id = get_ids("test", documents)

    train_docs = get_text(train_docs_id)
    test_docs = get_text(test_docs_id)

    train_categories = get_categories(train_docs_id)
    test_categories = get_categories(test_docs_id)

    return train_docs, test_docs, train_categories, test_categories, train_docs_id


def get_ids(t, documents):
    return list(filter(lambda doc: doc.startswith(t) and len(reuters.raw(doc)) > 100, documents))


def get_text(ids):
    return [reuters.raw(doc_id) for doc_id in ids]


def get_categories(ids):
    return [reuters.categories(doc_id) for doc_id in ids]
