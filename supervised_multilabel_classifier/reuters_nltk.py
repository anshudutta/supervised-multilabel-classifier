from nltk.corpus import reuters


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
