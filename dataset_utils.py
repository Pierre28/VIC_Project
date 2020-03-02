import numpy as np
import cv2


class DatasetUtils:

    def __init__(self, dataset_name, X, Y):
        self.dataset_name = dataset_name
        self.X = X
        self.Y = Y
        self.H, self.W = self.X[0].shape

    def sample_around_point(self, img, x, y, m):
        assert m > self.k
        pass

    def create_point_stat_model(self, points, k, images):
        """
        Creates a vector representation of the point using its neighbourhood.
        :return:
        """
        pass


    def transform_img_with_respect_to_stat(self, strategy):
        if strategy is None:
            X = self.X
        else:
            X = self.apply_filter(self.X, strategy)
        return X


    def create_lm_stat_model(self, k, strategy=None):
        self.k = k
        X = self.transform_img_with_respect_to_stat(strategy)
        stat_model = []
        for i in range(self.Y.shape[1]):
            stat_model.append(self.create_point_stat_model(points=self.Y[:, i, :], k=k, images=X))
        return stat_model

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


class KaggleDatasetUtils(DatasetUtils):
    
    def __init__(self, X, Y):
        super(KaggleDatasetUtils, self).__init__(dataset_name="Kaggle", X=X, Y=Y)

    def sample_around_point(self, img, x, y, m):
        """
        Since normal of point is not possible in Kaggle dataset, (landmark being too sparse), we simply sample in eight
        directions around a point.
        """
        print("Make sure sampling is performed using same strategy as training.")
        assert m > self.k
        center = self.get_neighbourhood(img, x, y, self.k)
        center /= np.linalg.norm(center)
        candidates = [center]
        for i in range(1, m-self.k+1):
            for u, v in [(x+i, y), (x-i, y), (x, y+i), (x, y-i), (x+i, y+i), (x-i, y+i), (x+i, y-i), (x-i, y-i)]:
                candidate = self.get_neighbourhood(img, u, v, self.k)
                candidate /= np.linalg.norm(candidate)
                candidates.append(candidate)
        return candidates

    def get_neighbourhood(self, img, x, y, k):
        out = np.zeros((2*k+1, 2*k+1))
        ngh = img[max(0, y - k):min(y + k + 1, self.W), max(0, x - k):min(x + k + 1, self.H)] # Carefull, x is horizontla, y vertical
        out[-min(0, y-k): 2*k+1 - max(0, y+k+1 -self.W), -min(0, x-k):2*k+1 - max(0, x+k+1-self.H)] = ngh
        return out

    def create_point_stat_model(self, points, k, images):

        point_model = []
        for index, img in enumerate(images):
            x, y = points[index]
            g = self.get_neighbourhood(img, int(x), int(y), k)
            g = g/np.linalg.norm(g)
            point_model.append(g.reshape((-1, )))

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
    fig = plt.figure()

    for i, (name, id) in enumerate(zip(names, ids)):
        plt.subplot(2, 2, i+1)
        mean, _ = stat_model[id]
        plt.imshow(mean.reshape((2*k+1, 2*k+1)))
        plt.title(f"{name}")
    plt.show()
    return fig


def check_sample_around_point(img, lm, id, name, samples, model, random=False):
    fig = plt.figure()
    plt.subplot(3, 3, 1)
    x, y = lm[id]
    plt.imshow(img, cmap="gray")
    plt.scatter([x], [y])
    plt.title(f"{name}")
    if random: np.random.shuffle(samples)
    for i, sample in enumerate(samples[-8:]):
        plt.subplot(3, 3, i+2)
        plt.imshow(sample, cmap="gray")
        dist = model.mahalanobis_dist(sample.reshape((-1,)), id)
        mse = np.linalg.norm(model.sm_means[id] - sample.reshape((-1,)))
        plt.title("D: {:.1e}, mse: {:.1e}".format(dist, mse) )
    plt.show()
    return fig

if __name__ == "__main__":

    import os
    from dataset import KaggleDataset
    path = os.path.join("data", "Kaggle")
    dataset = KaggleDataset(path)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = dataset.load(max_index=100)

    dataset_util = KaggleDatasetUtils(X_train, Y_train)
    k, strategy = 5, "Laplacian"
    stat_model = dataset_util.create_lm_stat_model(k, strategy=strategy)


    # Display stat models for different points :
    left_eye_corner_index, bottom_mouth_tip_index, nose_index, center_right_eye_index = 3, 13, 10, 1
    ids = [left_eye_corner_index, bottom_mouth_tip_index, nose_index, center_right_eye_index]
    names = ["left_eye_corner", "bottom_mouth_tip", "nose", "right_center_eye"]

    import matplotlib.pyplot as plt
    plot_unique_landmark(X_train[0], Y_train[0], ids, names)
    plot_point_stat_model(stat_model, ids, names)


    from model import ASM
    model = ASM()
    sm_means, sm_inv_covs = dataset_util.format_stat_model(stat_model)
    model.sm_means, model.sm_inv_covs = sm_means, sm_inv_covs
    X_transformed = dataset_util.transform_img_with_respect_to_stat(strategy=strategy)
    samples = dataset_util.sample_around_point(X_transformed[0], int(Y_train[0][ids[0]][0]), int(Y_train[0][ids[0]][1]), 15)
    check_sample_around_point(X_train[0], Y_train[0], ids[0], names[0], samples, model, random=True)

    print('.')