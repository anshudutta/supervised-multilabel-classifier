import os
import gensim


def load_model(limit=None):
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 'model'))
    raw_model_path = os.path.join(model_path, 'GoogleNews-vectors-negative300.bin')
    norm_model_path = os.path.join(model_path, 'GoogleNews-vectors-gensim-normed.bin')

    if not os.path.isfile(norm_model_path):
        if limit is None:
            raw_model = gensim.models.KeyedVectors.load_word2vec_format(raw_model_path, binary=True)
        else:
            raw_model = gensim.models.KeyedVectors.load_word2vec_format(raw_model_path, binary=True, limit=limit)
        raw_model.save(norm_model_path)
    model = gensim.models.KeyedVectors.load(norm_model_path, mmap='r')
    model.vectors_norm = model.vectors
    return model
