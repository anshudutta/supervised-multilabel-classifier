from abc import ABC, abstractmethod


class DataLoader(ABC):

    @abstractmethod
    def get_vectors(self, x_vec, y_vec):
        pass

    def get_vec_to_id(self, x_train, ids):
        vec2id = []
        for idx, x in enumerate(x_train):
            vec2id.append((x, ids[idx]))
        return vec2id
