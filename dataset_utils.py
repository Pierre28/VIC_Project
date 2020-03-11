import numpy as np
import cv2
from numpy_sift import SIFTDescriptor

class DatasetUtils:

    def __init__(self, dataset_name, X, Y):
        self.dataset_name = dataset_name
        self.X = X
        self.Y = Y
        self.H, self.W, *_ = self.X[0].shape

    def create_lm_stat_model(self, k, strategy=None, descriptor = None):
        self.SD = SIFTDescriptor(patchSize=2*k+1)
        self.k = k
        X = self.transform_img_with_respect_to_strat(self.X, strategy)
        stat_model = []
        for i in range(self.Y.shape[1]):
            stat_model.append(self.create_point_stat_model(points=self.Y[:, i, :], k=k, images=X, descriptor=descriptor))
        return stat_model

    def transform_img_with_respect_to_strat(self, data, strategy):
        pass

    @staticmethod
    def apply_filter(images: np.array, strategy):
        if strategy == "Laplacian":
            filter_fn = lambda x: cv2.Laplacian(x, cv2.CV_64F)
        elif strategy == "all_dir":
            kernel = np.array([[1, 2, 1], [2, -12, 2], [1, 2, 1]])
            filter_fn = lambda x: cv2.filter2D(x, -1, kernel)
        elif strategy == "sobel_x":
            kernel = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]])
            filter_fn = lambda x: cv2.filter2D(x, -1, kernel)
        else:
            raise NotImplementedError

        out = []
        for img in images:
            filtered = filter_fn(img)
            out.append(filtered)
        return out

    def sample_around_point(self, img, x, y, m):
        """
        Since normal of point is not possible in Kaggle dataset, (landmark being too sparse), we simply sample in eight
        directions around a point.
        """
        # print("Make sure sampling is performed using same strategy as training.")
        assert self.k, "First build statistical model"
        assert m > self.k
        center = self.get_neighbourhood(img, int(x), int(y), self.k)
        center /= np.linalg.norm(center)
        candidates = [(x, y, center)]
        for i in range(1, m-self.k+1):
            for u, v in [(x+i, y), (x-i, y), (x, y+i), (x, y-i), (x+i, y+i), (x-i, y+i), (x+i, y-i), (x-i, y-i)]:
                candidate = self.get_neighbourhood(img, int(u), int(v), self.k)
                candidate /= np.linalg.norm(candidate)
                candidates.append((u, v, candidate))
        return candidates

    def get_neighbourhood(self, img, x, y, k):
        out = np.zeros((2*k+1, 2*k+1))
        ngh = img[max(0, y - k):max(0, min(y + k + 1, self.W)), max(0, x - k):max(0, min(x + k + 1, self.H))] # Carefull, x is horizontla, y vertical
        out[-min(0, y-k): max(0, 2*k+1 - max(0, y+k+1 -self.W)), -min(0, x-k):max(0, 2*k+1 - max(0, x+k+1-self.H))] = ngh
        return out

    def create_point_stat_model(self, points, k, images, descriptor=None):

        point_model = []
        for index, img in enumerate(images):
            if index == 111: # Remove 111 index in iBug.
                continue
            x, y = points[index]
            g = self.get_neighbourhood(img, int(x), int(y), k)
            if descriptor == "sift":
                g = self.SD.describe(g)
            else:
                if np.linalg.norm(g) == 0:
                    continue
                g = g/np.linalg.norm(g)
                g = g.reshape((-1, ))
            point_model.append(g)

        mean = np.mean(point_model, axis=0)
        cov = np.cov(np.array(point_model).T)
        return mean, cov

    @staticmethod
    def format_stat_model(model):
        """
        Input : Stat model (list of (mean, cov) for each lm point)
        :return: Tuple (means, invs), with
        means : list of lm means
        invs : list of inverted covs
        """
        means, invs = [], []
        for m, c in model:
            means.append(m)
            invs.append(np.linalg.inv(c))
        return means, invs

class iBug300DatasetUtils(DatasetUtils):
    def __init__(self, X, Y):
        super(iBug300DatasetUtils, self).__init__(dataset_name="300W", X=X, Y=Y)

    def transform_img_with_respect_to_strat(self, data, strategy):
        X = np.array([cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in data])
        if strategy is None:
            pass
        else:
            X = self.apply_filter(X, strategy)
        return X

class KaggleDatasetUtils(DatasetUtils):
    
    def __init__(self, X, Y):
        super(KaggleDatasetUtils, self).__init__(dataset_name="Kaggle", X=X, Y=Y)

    def transform_img_with_respect_to_strat(self, data, strategy):
        if strategy is None:
            X = data
        else:
            X = self.apply_filter(data, strategy)
        return X

def plot_unique_landmark(img, lm, ids, names):
    fig = plt.figure()
    for i, (name, id) in enumerate(zip(names, ids)):
        plt.subplot(2, 2, i+1)
        x, y = lm[id]
        plt.imshow(img, cmap="gray")
        plt.scatter([x], [y])
        plt.title(f"{name}")
    plt.show()
    return fig


def plot_point_stat_model(stat_model, ids, names):
    fig = plt.figure(figsize=(10, 10))

    for i, (name, id) in enumerate(zip(names, ids)):
        plt.subplot(2, 2, i+1)
        mean, _ = stat_model[id]
        plt.imshow(mean.reshape((2*k+1, 2*k+1)), cmap="gray")
        plt.title(f"{name}", size="xx-large")
        plt.xticks([])
        plt.yticks([])

    plt.subplots_adjust(wspace=0.025, hspace=0.1)
    plt.savefig("fig/none_local.png",  bbox_inches='tight')
    plt.show()
    return fig


def check_sample_around_point(img, lm, id, name, samples, model, random=False, sorted_=False, dist="mah"):
    fig = plt.figure()
    plt.subplot(3, 3, 1)
    x, y = lm[id]
    # plt.imshow(img, cmap="gray")
    # plt.scatter([x], [y])
    reshape_size = int(np.sqrt(model.sm_means[id].shape[0]))
    plt.imshow(model.sm_means[id][:121].reshape((reshape_size, reshape_size)), cmap="gray")
    plt.title(f"{name}")
    if random: np.random.shuffle(samples)
    dist_fn = model.mahalanobis_dist if dist == "mah" else model.euclidean_distance
    if sorted_:
        sorted_index = np.argsort([dist_fn(sample.reshape((-1,)), id) for u, v, sample in samples])
        samples = np.array(samples)[sorted_index]
    for i, (u, v, sample) in enumerate(samples[-8:]):
        plt.subplot(3, 3, i+2)
        plt.imshow(sample, cmap="gray")
        dist = dist_fn(sample.reshape((-1,)), id)
        plt.title("{}, {}:{:.1e}".format(u, v, dist))

    print("{}, {}".format(x, y))
    plt.show()
    return fig

if __name__ == "__main__":

    import os
    from dataset import KaggleDataset, iBug300WDataset
    dataset_name = "Kaggle"
    path = os.path.join("data", dataset_name)
    # dataset = KaggleDataset(path) #KaggleDataset(path) #
    dataset_name = "300W"
    path = os.path.join("data", dataset_name, "01_Indoor")
    dataset = iBug300WDataset(path)

    X_train, X_val, X_test, Y_train, Y_val, Y_test = dataset.load() # For kaglle
    X_train, X_val, X_test, Y_train, Y_val, Y_test = dataset.load_cropped_resized()

    k, strategy, descriptor = 12, None, "sift"
    dataset_util = KaggleDatasetUtils(X_train, Y_train)
    # stat_model = dataset_util.create_lm_stat_model(k, strategy=strategy)

    iBug_utils = iBug300DatasetUtils(X_train, Y_train)
    stat_model = iBug_utils.create_lm_stat_model(k, strategy=strategy, descriptor=descriptor)

    dataset_util = iBug_utils if iBug_utils is not None else dataset_util

    # Display stat models for different points :
    left_eye_corner_index, bottom_mouth_tip_index, nose_index, center_right_eye_index = 10, 20, 30, 40
    ids = [left_eye_corner_index, bottom_mouth_tip_index, nose_index, center_right_eye_index]
    names = ["left_eye_corner", "bottom_mouth_tip", "nose", "right_center_eye"]

    import matplotlib.pyplot as plt
    plot_unique_landmark(X_train[0], Y_train[0], ids, names)
    plot_point_stat_model(stat_model, ids, names)


    from model import ASM
    model = ASM()
    sm_means, sm_inv_covs = dataset_util.format_stat_model(stat_model)
    model.sm_means, model.sm_inv_covs = sm_means, sm_inv_covs
    X_transformed = dataset_util.transform_img_with_respect_to_strat(X_train, strategy=strategy)
    samples = dataset_util.sample_around_point(X_transformed[0], int(Y_train[0][ids[2]][0]), int(Y_train[0][ids[2]][1]), 15)
    if descriptor is not None:
        samples_ = model.sift_transform(samples, patch_size=25)
        samples = [s[-1] for s in samples_]
    check_sample_around_point(X_train[0], Y_train[0], ids[2], names[2], samples, model, random=False, sorted_=True, dist="mse")

    print('.')


