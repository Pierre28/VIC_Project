from sklearn.decomposition import PCA
import numpy as np
from utils import pointwise_euclidean_distance
from numpy_sift import SIFTDescriptor
from shape_aligner import Shape


class BaseDetector:
    """
    Base class for landmark model detector
    """

    def __init__(self):
        pass

    def fit(self, X, Y):
        """
        Fitting the model to the training data
        """
        pass

    def predict(self, X):
        """
        Predict a set of landmark points onto a new image X
        """
        pass

class ASM(BaseDetector):
    """
    Active Shape Model implementation : https://www.face-rec.org/algorithms/AAM/app_models.pdf
    """


    def __init__(self, modes=20):
        """
        :param modes: Number of eigenvectors to retain
        """
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
        """
        Align dataset to a common co-ordinate frame using Procrustes Analysis. Then perform PCA.
        :param Y: Set of landmarks
        :param pca_components: Number of egeinvectors to retain
        :return:
        """
        self.shapes = [Shape.from_vector(y) for y in Y]
        self.w = self.compute_weight_matrix(Y)
        self.aligned_shapes = self.__procrustes(self.shapes)

        Y = np.array([shape.get_vector() for shape in self.aligned_shapes])
        if pca_components:
            assert pca_components < Y.shape[1]
            pca = PCA(n_components=pca_components)
            self.deformable_model = pca.fit_transform(Y).mean(axis=0)
            self.deformable_model_set = pca.transform(Y)
            self.dm_vp = pca.explained_variance_
            self.model_explained_var = sum(pca.explained_variance_ratio_)
            self.P = pca.components_
            self.mean_ = pca.mean_
        else:
            raise NotImplementedError

    def get_mean_shape(self, shapes:list) -> list:
        """
        Util function to get the mean of shapes
        """
        s = shapes[0]
        for shape in shapes[1:]:
            s = s + shape
        return s.__div__(len(shapes))

    def __procrustes(self, shapes):
        """ This function aligns all shapes passed as a parameter by using
        Procrustes analysis
        From : https://github.com/YashGunjal/asm

        :param shapes: A list of Shape objects
        """
        # First rotate/scale/translate each shape to match first in set
        shapes[1:] = [s.align_to_shape(shapes[0], self.w) for s in shapes[1:]]

        # Keep hold of a shape to align to each iteration to allow convergence
        a = shapes[0]
        trans = np.zeros((4, len(shapes)))
        converged = False
        current_accuracy = np.inf
        while not converged:
            # Now get mean shape
            mean = self.get_mean_shape(shapes)
            # Align to shape to stop it diverging
            mean = mean.align_to_shape(a, self.w)
            # Now align all shapes to the mean
            for i in range(len(shapes)):
                # Get transformation required for each shape
                trans[:, i] = shapes[i].get_alignment_params(mean, self.w)
                # Apply the transformation
                shapes[i] = shapes[i].apply_params_to_shape(trans[:, i])

            # Test if the average transformation required is very close to the
            # identity transformation and stop iteration if it is
            accuracy = np.mean(np.array([1, 0, 0, 0]) - np.mean(trans, axis=1)) ** 2
            # If the accuracy starts to decrease then we have reached limit of precision
            # possible
            if accuracy > current_accuracy:
                converged = True
            else:
                current_accuracy = accuracy
        return shapes

    @staticmethod
    def mean_shape(shapes:np.array) -> np.array:
        """
        Util function to get mean of a shape
        """
        mean_ = np.mean(shapes, axis=0)
        return mean_

    @staticmethod
    def compute_weight_matrix(shapes):
        """
        Compute the weight matrix of shapes used in Procrustes Analysis
        """
        assert shapes.ndim == 3, "Must not be flatten first"
        num_img, num_pts, _ = shapes.shape

        euclidean_dist = lambda x, y: np.sqrt(np.sum(np.power(x - y, 2)))

        # Compute pointwise distances
        inner_point_dist = np.zeros((num_img, num_pts, num_pts))
        for s, shape in enumerate(shapes):
            for i in range(num_pts):
                for j in range(num_pts):
                    inner_point_dist[s, i, j] = euclidean_dist(shape[i, :], shape[j, :])

        # Normalize distance by variance across shapes
        weight = np.zeros(num_pts)
        for i in range(num_pts):
            for j in range(num_pts):
                weight[i] += np.var(inner_point_dist[:, i, j])

        inv_weight = 1/weight
        return inv_weight

    def generate_lm(self, mode=0, scale=10):
        """
        Generate landmark by varying a given mode
        """
        assert self.deformable_model is not None, "You must build the model first"
        limit = np.sqrt(3)*np.sqrt(self.dm_vp)[mode]/scale
        dm = self.deformable_model.copy()
        delta = np.random.uniform(-limit, limit)
        dm[mode] += delta
        lm = dm @ self.P + self.mean_
        return lm.reshape((-1, 2))

    def create_mode_map(self, mode=0, n=5):
        """
        Create a map of a mode by varying its value onto the whole plausible range
        """
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
        """
        Utils function implementing the Mahalanobis distance
        """
        assert g.ndim == 1, "Must flatten neighbourhood before computing distance"
        dist = (g - self.sm_means[i]).T @ self.sm_inv_covs[i] @ (g - self.sm_means[i])
        return dist

    def euclidean_distance(self, g, i):
        """
        Utils function implementing the Euclidean distance
        """
        dist = (g - self.sm_means[i]).T @ np.eye(self.sm_inv_covs[i].shape[0]) @ (g - self.sm_means[i])
        return dist


    def predict(self, X):
        """
        Generate landmark for image X using mean shape model
        """
        lm = self.generate_lm()
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

    def apply_constraints(self, b):
        """
        Apply constraint on shape vector
        """
        limits = 3* np.sqrt(self.dm_vp)
        b = np.clip(b, -limits, limits)
        return b

    @staticmethod
    def sift_transform(patches, patch_size):
        """
        Util function to compute SIFT descriptor on patch
        """
        SD = SIFTDescriptor(patchSize = patch_size)
        out = []
        for x, y, patch in patches:
            patch_ = SD.describe(patch)
            out.append((x, y, patch_))
        return out

    def iteration(self, dataset_processor, X, img, m, dist, descriptor, patch_size):
        """
        Perform a single iteration of the ASM algorithm. Specifically:
        1. Initiate
        2. Find best candidate around current shape
        3. Find best parameters to align to the new shape
        4. Project back to image space
        :param dataset_processor: instance of DatasetUtils
        :param X: np.array, landmark prior to iteration
        :param img: np.array, image to fit landmark to
        :param m: search range around each point. Search will be performed on m-k, where k is the descriptor size
        :param dist: Fit function
        :param descriptor: "sift" or None
        :param patch_size: Patch size used for the descriptor
        :return:
        """

        # Initiate
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

        old_s = Shape.from_vector(X_)

        # Align & update params - Navigating form Image space - Image co-ordinate - Model space
        new_s = old_s.align_to_shape(Shape.from_vector(self.mean_), self.w)
        diff = new_s.get_vector() - self.mean_
        new_shape_vector = self.mean_.copy()
        b = self.P @ diff
        b_ = self.apply_constraints(b)
        new_shape_vector += b_ @ self.P
        new_shape_aligned = Shape.from_vector(new_shape_vector).align_to_shape(old_s, self.w)
        new_shape_vector_aligned = new_shape_aligned.get_vector()

        return new_shape_vector_aligned.reshape((-1, 2))


    def asm(self, dataset_processor, img, m, max_it=10, eps=1, dist="mah", descriptor="sift", patch_size=15, return_hist=None):
        """
        Perform Active Shape Model on image to fit a landmark. Iterativily calls iteration, while update stays above a given
        threshold
        :param dataset_processor:  instance of DatasetUtils
        :param img:  np.array, image to fit landmark to
        :param m: search range around each point. Search will be performed on m-k, where k is the descriptor size
        :param max_it: maximum number of iteration to perform. ASM classicaly converges in a few epochs
        :param eps: threshold to assert convergence
        :param dist: fit function
        :param descriptor: "sift" or None
        :param patch_size: Patch size of the descriptor
        :param return_hist: Bool, whether to store each iteration solution or not
        :return: fitted landmark
        """
        X = self.mean_.copy().reshape((-1, 2))
        if return_hist: lm_hist = [X]
        converge, it = False, 0
        while not converge and it < max_it:
            it += 1
            X_ = self.iteration(dataset_processor, X, img, m, dist=dist,  descriptor=descriptor, patch_size=patch_size)
            if np.linalg.norm(X - X_) < eps:
                converge = True
            X = X_
            if return_hist: lm_hist.append(X)

        if return_hist: return X, lm_hist
        return X

if __name__ == "__main__":

    # Used to debug

    import os
    # from dataset import KaggleDataset
    # path = os.path.join("data", "Kaggle")
    # dataset = KaggleDataset(path)
    from dataset import iBug300WDataset
    path = os.path.join("data", "300W", "01_Indoor")
    dataset = iBug300WDataset(path)
    # X_train, X_val, X_test, Y_train, Y_val, Y_test = dataset.load(max_index=100)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = dataset.load_cropped_resized()
    model = ASM()
    model.fit(X_train, Y_train)


    from dataset_utils import KaggleDatasetUtils
    dataset_processor = KaggleDatasetUtils(X_train, Y_train)
    strategy, dist, k = "all_dir", "mse", 10
    lm_stat_model = dataset_processor.create_lm_stat_model(k=k, strategy=strategy)
    sm_means, sm_inv_covs = dataset_processor.format_stat_model(lm_stat_model)

    model.sm_means, model.sm_inv_covs = sm_means, sm_inv_covs

    # img, lm = X_test[3], Y_test[3]
    import matplotlib.pyplot as plt

    im_id = 10
    before_mean_ = model.mean_
    X_test_filtered = dataset_processor.transform_img_with_respect_to_strat(X_test, strategy=strategy)
    X_ = model.asm(dataset_processor, X_test_filtered[im_id], 15, 10, dist=dist)
    # X_, b, Xt, Yt, s, theta = model.asm_algo(dataset_processor, X_test_filtered[im_id], 15, 10, dist=dist)

    from visualizer import display_lm, draw_mode_variation_single_plot, draw_mode_variation
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

    for i in range(20):
        display_lm(X_val[i], Y_val[i])