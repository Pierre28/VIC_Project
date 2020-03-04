from sklearn.decomposition import PCA
import numpy as np
from utils import pointwise_euclidean_distance
from numpy_sift import SIFTDescriptor


class BaseDetector:

    def __init__(self):
        pass

    def fit(self, X, Y):
        pass

    def predict(self, X):
        pass

    def load(self):
        pass

    def save(self):
        pass

class ASM(BaseDetector):

    def __init__(self, modes=20):
        super(ASM, self).__init__()
        self.deformable_model = None
        self.modes = modes
        self.Xt, self.Yt = None, None # Translation param
        self.s = None # Scale param
        self.theta = None  # Pose param


        self.sm_means, self.sm_inv_covs = None, None


    def fit(self, X, Y):

        self.build_model(Y, pca_components=self.modes)

    def build_model(self, Y, pca_components=None):
        if pca_components:
            assert pca_components < Y.shape[1]*Y.shape[2]
            pca = PCA(n_components=pca_components)
            self.deformable_model = pca.fit_transform(Y.reshape(Y.shape[0], -1)).mean(axis=0)
            self.deformable_model_set = pca.transform(Y.reshape(Y.shape[0], -1))
            self.dm_vp = pca.explained_variance_
            self.model_explained_var = sum(pca.explained_variance_ratio_)
            self.P = pca.components_
            self.mean_ = pca.mean_
        else:
            raise NotImplementedError

    def generate_lm(self, mode=0, scale=10):
        assert self.deformable_model is not None, "You must build the model first"
        limit = np.sqrt(3)*np.sqrt(self.dm_vp)[mode]/scale
        dm = self.deformable_model.copy()
        delta = np.random.uniform(-limit, limit)
        dm[mode] += delta
        lm = dm @ self.P + self.mean_
        return lm.reshape((-1, 2))

    def create_mode_map(self, mode=0, n=5):
        assert self.deformable_model is not None, "You must build the model first"
        limit = np.sqrt(3)*np.sqrt(self.dm_vp)[mode]
        variations = np.linspace(-limit, +limit, n)
        lms = []
        for var in variations:
            dm = self.deformable_model_set[0].copy()
            dm[mode] += var
            lm = dm @ self.P + self.mean_
            lms.append(lm.reshape((-1, 2)))
        return lms, variations

    def mahalanobis_dist(self, g, i):
        assert g.ndim == 1, "Must flatten neighbourhood before computing distance"
        dist = (g - self.sm_means[i]).T @ self.sm_inv_covs[i] @ (g - self.sm_means[i])
        return dist

    def euclidean_distance(self, g, i):
        dist = (g - self.sm_means[i]).T @ np.eye(self.sm_inv_covs[i].shape[0]) @ (g - self.sm_means[i])
        return dist


    def fit_fn(self):
        pass

    def predict(self, X):
        lm = self.generate_lm()
        # TODO: iteratively update fit function
        return np.array([lm for _ in X])

    @staticmethod
    def T(Xt, Yt, s, theta, x: np.array) -> np.array:
        assert x.ndim == 2, "Reshape to landmark (n_lm, 2) before transforming"
        """
        Applies translation - rotation - scale to a model instance. Convention : capital letter : image coordinate, lower case : object-aligned coordinates
        :param Xt: float, trans x
        :param Yt: float, trans y
        :param s: float, scale
        :param theta: float, pose
        :param x: np.array : (lm_size, 2)
        :return: np.array: (lm_size, 2)
        """

        t = np.array([[Xt, Yt]])
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        Y = t + s * x @ R
        return Y

    @staticmethod
    def T_inv(Xt, Yt, s, theta, Y: np.array) -> np.array:
        assert Y.ndim == 2, "Reshape to landmark (n_lm, 2) before transforming"
        """
        Invert of T
        """
        t = np.array([Xt, Yt])
        R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        x = (Y - t)/s @ R
        return x

    @staticmethod
    def get_pose_parameters(x, y): # TODO : complete
        assert x.ndim == 2, "Reshape to landmark (n_lm, 2) before transforming"
        assert y.ndim == 2, "Reshape to landmark (n_lm, 2) before transforming"
        """
        Compute best matching pose parameters matching model x to target points y
        :return tuple : (Xt, Yt, s, theta)
        """
        return 0, 0, 1, 0

    def match_model_to_target_points(self, Y, max_it=10, eps=1e-3): # Protocol 1, page 9 (https://pdfs.semanticscholar.org/ebc2/ceba03a0f561dd2ab27c97b641c649c48a14.pdf)
        # Initialize model
        b = np.zeros((self.modes,))
        Xt, Yt, s, theta = None, None, None, None
        # Main loop
        converge, it = False, 0
        while not converge and it < max_it:
            it += 1
            x = self.mean_ + b @ self.P
            x_reshaped, Y_reshaped = x.reshape((-1, 2)), Y.reshape((-1, 2))
            Xt, Yt, s, theta = self.get_pose_parameters(x_reshaped, Y_reshaped)
            y = self.T_inv(Xt, Yt, s, theta, Y_reshaped) # Project Y into the model co-ordinate frame
            y = y.reshape((-1,))
            y = y / (y @ self.mean_) # Project y into the tangent plane to x ̄ by scaling #TODO : Seems unappropriate.
            b_ = self.P @ (y - self.mean_) # Update the model parameters to match to y′
            b_ = self.apply_constraints(b_) # Apply constraints on b
            if np.abs(b - b_).sum() < eps:
                converge=True
            b = b_.copy()
        return b, Xt, Yt, s, theta, converge, it

    def apply_constraints(self, b):
        limits = 3* np.sqrt(self.dm_vp)
        b = np.clip(b, -limits, limits)
        return b

    @staticmethod
    def sift_transform(patches, patch_size):
        SD = SIFTDescriptor(patchSize = patch_size)
        out = []
        for x, y, patch in patches:
            patch_ = SD.describe(patch)
            out.append((x, y, patch_))
        return out

    def iteration(self, dataset_processor, X, img, m, dist, descriptor, patch_size):
        X_ = np.zeros_like(X)

        dist_fn = self.euclidean_distance if dist == "mse" else self.mahalanobis_dist

        # Find best candidate around current shape
        for i, (x, y) in enumerate(X):
            candidates = dataset_processor.sample_around_point(img, x, y, m)
            if descriptor=="sift": candidates = self.sift_transform(candidates, patch_size=patch_size)
            u_, v_, min_dist = None, None, np.inf
            for u, v, candidate in candidates:
                dist = dist_fn(candidate.reshape((-1,)), i)
                if dist < min_dist:
                    min_dist = dist
                    u_, v_ = u, v
            X_[i] = [u_, v_]

        # Align to X_
        # b = self.P @ X_.reshape((-1, ))
        # b_ = self.apply_constraints(b)
        # new_shape = b_ @ self.P
        diff = X_.reshape((-1, )) - self.mean_
        new_shape = self.mean_.copy()
        b = self.P @ diff
        b_ = self.apply_constraints(b)
        new_shape += b_ @ self.P
        return new_shape.reshape((-1, 2))

    def asm(self, dataset_processor, img, m, max_it=10, eps=1, dist="mah", descriptor="sift", patch_size=15):

        X = self.mean_.copy().reshape((-1, 2))
        converge, it = False, 0
        while not converge and it < max_it:
            it += 1
            X_ = self.iteration(dataset_processor, X, img, m, dist=dist,  descriptor=descriptor, patch_size=patch_size)
            if np.linalg.norm(X - X_) < eps:
                converge = True
            X = X_

        return X

    def asm_algo(self, dataset_processor, img, m, max_it=10, eps=1, dist="mah"):

        X  = (self.mean_ + self.deformable_model @ self.P).reshape((-1, 2))
        X_ = np.zeros_like(X)

        dist_fn = self.euclidean_distance if dist == "mse" else self.mahalanobis_dist
        converge, it = False, 0
        while not converge and it < max_it:
            it += 1
            for i, (x, y) in enumerate(X):
                candidates = dataset_processor.sample_around_point(img, x, y, m)
                u_, v_, min_dist = None, None, np.inf
                for u, v, candidate in candidates:
                    dist = dist_fn(candidate.reshape((-1,)), i)
                    if dist < min_dist:
                        min_dist = dist
                        u_, v_ = u, v
                X_[i] = [u_, v_]
            b, Xt, Yt, s, theta, *_ = self.match_model_to_target_points(X_)

            # Generate b
            x = b @ self.P  # + self.mean_
            X_ = self.T(Xt, Yt, s, theta, x.reshape((-1, 2)))

            if np.linalg.norm(X - X_) < eps:
                converge = True
            X = X_.copy()

        return X, b, Xt, Yt, s, theta


def test_T_fn(model):
    def T_T_inv_test(model):
        x = model.generate_lm()
        Y = model.T(1, 1, 2, np.pi / 2, x)
        X = model.T_inv(1, 1, 2, np.pi / 2, Y)
        return ~np.any(X-x)

    def T_id(model):
        x = model.generate_lm()
        y = model.T(0, 0, 1, 0, x)
        return ~np.any(y - x)

    def Tinv_id(model):
        x = model.generate_lm()
        y = model.T_inv(0, 0, 1, 0, x)
        return ~np.any(y - x)

    assert T_T_inv_test(model)
    assert T_id(model)
    assert Tinv_id(model)

def test_matching_model(model):
    b, Xt, Yt, s, theta, cv, it = model.match_model_to_target_points(model.mean_)
    # assert it==1, "Inputing normalized mean_ of model should return a null shape"
    return

if __name__ == "__main__":

    import os
    from dataset import KaggleDataset
    path = os.path.join("data", "Kaggle")
    dataset = KaggleDataset(path)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = dataset.load(max_index=1000)
    model = ASM()
    model.fit(X_train, Y_train)

    test_T_fn(model)
    test_matching_model(model)

    from dataset_utils import KaggleDatasetUtils
    dataset_processor = KaggleDatasetUtils(X_train, Y_train)
    strategy, dist, k = "all_dir", "mse", 10
    lm_stat_model = dataset_processor.create_lm_stat_model(k=k, strategy=strategy)
    sm_means, sm_inv_covs = dataset_processor.format_stat_model(lm_stat_model)

    model.sm_means, model.sm_inv_covs = sm_means, sm_inv_covs


    import matplotlib.pyplot as plt
    img, lm = X_test[3], Y_test[3]
    def display_lm(img, lm, lm_gt=None, legend=False):
        plt.imshow(img, cmap="gray")
        plt.scatter(lm[:, 0], lm[:,1], c="r", label="pred")
        if lm_gt is not None :
            plt.scatter(lm_gt[:, 0], lm_gt[:, 1], c="g", label="gt")
            plt.title("PED : {}".format(pointwise_euclidean_distance(lm, lm_gt)))
        if legend: plt.legend()
        plt.show()

    def display_candidate(cand):
        plt.imshow(cand, cmap="gray")
        plt.show()

    def draw_mode_variation(img, lms, variations, mode):
        fig = plt.figure()
        n = int(np.sqrt(len(lms)))
        for i, (lm, var) in enumerate(zip(lms[:n**2], variations[:n**2])):
            plt.subplot(n, n, i+1)
            plt.imshow(img)
            plt.scatter(lm[:, 0], lm[:, 1])
            plt.title("{:.1f}".format(var))
        plt.title("Mode : {}".format(mode))
        plt.show()
        return fig


    def draw_mode_variation_single_plot(img, lms, variations, mode):
        fig = plt.figure()
        n = int(np.sqrt(len(lms)))
        plt.subplot(1, 1, 1)
        plt.imshow(img, cmap="gray")
        for i, (lm, var) in enumerate(zip(lms[:n ** 2], variations[:n ** 2])):
            plt.scatter(lm[:, 0], lm[:, 1], label="{:.1f}".format(var))
        plt.legend()
        plt.title("Mode : {}".format(mode))
        plt.show()
        return fig


    im_id = 10
    before_mean_ = model.mean_
    X_test_filtered = dataset_processor.transform_img_with_respect_to_strat(X_test, strategy=strategy)
    X_ = model.asm(dataset_processor, X_test_filtered[im_id], 15, 10, dist=dist)
    # X_, b, Xt, Yt, s, theta = model.asm_algo(dataset_processor, X_test_filtered[im_id], 15, 10, dist=dist)

    # Before fitting :
    X = model.mean_.reshape((-1, 2))
    display_lm(X_test_filtered[im_id], X, Y_test[im_id], legend=True)

    # After fitting :
    after_mean_ = model.mean_
    display_lm(X_test_filtered[im_id], X_, Y_test[im_id], legend=True)


    # Let's play with the shapes
    mode = 6 # mode 3 nez bouche mode 4 ouverture bouche mode 5 yeux
    lms, variations = model.create_mode_map(mode, 9)
    draw_mode_variation(np.zeros_like(X_train[0]), lms, variations, mode=mode)
    draw_mode_variation_single_plot(np.zeros_like(X_train[0]), lms, variations, mode=mode)
