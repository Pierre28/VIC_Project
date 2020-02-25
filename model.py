from sklearn.decomposition import PCA
import numpy as np

class BaseDetector:

    def __init__(self):
        pass

    def fit(self, X, Y):
        pass

    def predict(self, X):
        pass

class ASM(BaseDetector):

    def __init__(self):
        super(ASM, self).__init__()
        self.deformable_model = None

    def fit(self, X, Y):

        self.build_model(Y, pca_components=20)



    def build_model(self, Y, pca_components=None):
        if pca_components:
            assert pca_components < Y.shape[1]*Y.shape[2]
            pca = PCA(n_components=pca_components)
            self.deformable_model = pca.fit_transform(Y.reshape(Y.shape[0], -1)).mean(axis=0)
            self.dm_vp = pca.explained_variance_
            self.model_explained_var = sum(pca.explained_variance_ratio_)
            self.P = pca.components_
            self.mean_ = pca.mean_
        else:
            raise NotImplementedError

    def generate_lm(self, mode=0):
        assert self.deformable_model is not None, "You must build the model first"
        limit = np.sqrt(3)*self.dm_vp[mode]/100
        dm = self.deformable_model.copy()
        delta = np.random.uniform(-limit, limit)
        dm[mode] += delta
        lm = dm @ self.P + self.mean_
        return lm.reshape((-1, 2))

    def fit_fn(self):
        pass

    def predict(self, X):
        lm = self.generate_lm()
        # TODO: iteratively update fit function
        return np.array([lm for _ in X])
