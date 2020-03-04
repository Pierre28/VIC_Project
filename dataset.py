import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

np.random.seed(0)

class Dataset:

    def __init__(self, path, img_extension):
        self.path = path
        self.img_extension = img_extension
        self.indexes = None
        self.len = None

    def load(self):
        """
        Load all images and corresponding landmarks
        :return: X (np.array) list of images, Y (np.array) list of landmarks
        """
        return None, None

    def read_test_train_split(self):
        with open(self.path + "/indices.txt", "r") as f:
            lines = f.readlines()
            indices = [[int(x) for x in line.rstrip().split(":")[1].split(",")] for line in lines]
            return indices


class iBug300WDataset(Dataset):
    """
    https://ibug.doc.ic.ac.uk/download/300VW_Dataset_2015_12_14.zip/
    """
    def __init__(self, path, img_extension="png"):
        super().__init__(path, img_extension)
        self._retrieve_index()
        self.len = len(self.indexes)

    def __repr__(self):
        return "Dataset : {} samples".format(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        if isinstance(item, int):
            index = self.indexes[item]
            img = self.load_image(index)
            landmark = self.load_landmark(index)
        elif isinstance(item, slice):
            print("Only slice using [start, stop, step], all written")
            start, stop, step = item.start, item.stop, item.step
            index = [self.indexes[i] for i in range(start, stop, step)]
            img = [self.load_image(i) for i in index]
            landmark = [self.load_landmark(i) for i in index]
        else:
            raise TypeError("index must be int or slice")
        return (index, img, landmark)

    def __iter__(self):
        pass


    def _retrieve_index(self):
        self.indexes = []
        for file_name in os.listdir(self.path):
            name_match = re.match("([a-zA-Z0-9\_]+).{}".format(self.img_extension), file_name)
            if name_match: self.indexes.append(name_match.group(1))


    def load_image(self, index:str, convert_color=False) -> np.array:
        """
        Note : cv2 use BRG convention. To switch de RGB, use cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        """
        _img = cv2.imread(os.path.join(self.path, index + "." + self.img_extension))
        if convert_color: _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        return _img

    def load_images(self):
        images = []
        for index in self.indexes:
            images.append(self.load_image(index, convert_color=True))
        return np.array(images)

    def load(self, train_test_split=True):
        X, Y = self.load_images(), self.load_landmarks()
        if train_test_split:
            train_indices, val_indices, test_indices = self.read_test_train_split()
            X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
            Y_train, Y_val, Y_test = X[train_indices], X[val_indices], X[test_indices]

            return X_train, X_val, X_test, Y_train, Y_val, Y_test
        else:
            return X, Y


    def load_cropped(self, offset=15, train_test_split=True):
        images = []
        landmarks = []
        for index in self.indexes:
            im = self.load_image(index, convert_color=True)
            lm = self.load_landmark(index)
            bbox = self.create_bbox(lm)
            images.append(self.crop_w_bbox(im, bbox, offset))
            landmarks.append(self.translate_landmark(lm, bbox, offset))
        X, Y =  np.array(images), np.array(landmarks)
        if train_test_split:
            train_indices, val_indices, test_indices = self.read_test_train_split()
            X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
            Y_train, Y_val, Y_test = Y[train_indices], Y[val_indices], Y[test_indices]

            return X_train, X_val, X_test, Y_train, Y_val, Y_test
        else:
            return X, Y

    @staticmethod
    def resize(img, size):
        img_ = cv2.resize(img, dsize=size)
        return img_

    @staticmethod
    def rescale_lm(lm, xscale, yscale):
        lm_ = np.zeros_like(lm)
        for i, (x, y) in enumerate(lm):
            lm_[i] = [x*yscale, y*xscale]
        return lm_

    def rescale_set(self, im_set, lm_set, size):
        out_im, out_lm = [] , []
        for im, lm in zip(im_set, lm_set):
            try :
                h, w, *_ = im.shape
                s_h, s_w = size[0] / h, size[1] / w
                img = self.resize(im, size=size)
                lm_ = self.rescale_lm(lm, s_h, s_w)
                out_im.append(img)
                out_lm.append(lm_)
            except:
                pass
        return [np.array(out_im), np.array(out_lm)]

    def load_cropped_resized(self, offset=15, size=(128, 128), train_test_split=True):
        if train_test_split:
            X_train, X_val, X_test, Y_train, Y_val, Y_test = self.load_cropped(offset=offset)
            rescale_im_sets, rescale_lm_sets = [], []
            for sets in [(X_train, Y_train), (X_val, Y_val), (X_test, Y_test)]:

                im_set, lm_set = self.rescale_set(sets[0], sets[1], size)
                rescale_im_sets.append(im_set)
                rescale_lm_sets.append(lm_set)

            X_train, X_val, X_test = rescale_im_sets
            Y_train, Y_val, Y_test = rescale_lm_sets
            return X_train, X_val, X_test, Y_train, Y_val, Y_test

        else:
            X, Y = self.load_cropped(offset=offset)
            raise NotImplementedError


    def load_landmark(self, index: str) -> np.array:
        with open(os.path.join(self.path, index + '.pts'), "r") as f:
            lines = f.readlines()
            _version = re.match("version: ([0-9]+)\n$", lines[0]).group(1)
            _n_points = re.match("n_points: ([0-9]{2}\n$)", lines[1]).group(1)
            _points = []
            for line in lines[3:-1]:
                match = re.match("([0-9]+.[0-9]+) ([0-9]+.[0-9]+)\n$", line)
                _points.append([float(match.group(1)), float(match.group(2))])
        return np.array(_points)

    def load_landmarks(self):
        landmarks = []
        for index in self.indexes:
            landmarks.append(self.load_landmark(index))
        return np.array(landmarks)

    def create_train_test_split(self):
        # Get number of indexes
        indices = np.arange(self.len)
        np.random.shuffle(indices)
        train_indices, val_indices, test_indices = np.split(indices, [int(self.len*.6), int(self.len*.8)])

        with open(self.path + "/indices.txt", "w") as f:
            train_str = "train_index:" + ",".join([str(x) for x in sorted(train_indices)]) + "\n"
            val_str = "val_index:" + ",".join([str(x) for x in sorted(val_indices)]) + "\n"
            test_str = "test_index:" + ",".join([str(x) for x in sorted(test_indices)]) + "\n"
            out_str = train_str + val_str + test_str
            f.write(out_str)

    @staticmethod
    def matplotlib_visualize_landmark(img, landmark, size=5, is_rgb=True):
        if not is_rgb: img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else: img_ = np.copy(img)

        fig, ax = plt.subplots(1)
        ax.scatter([x[0] for x in landmark], [x[1] for x in landmark], c="b")
        ax.imshow(img_)
        return fig

    @staticmethod
    def create_bbox(landmark):
        (xmin, ymin), (xmax, ymax) = np.min(landmark, axis=0), np.max(landmark, axis=0)
        return (xmin, ymin, xmax, ymax)

    @staticmethod
    def matplotlib_visualize_bbox(img, bbox, is_rgb=True):
        if not is_rgb: img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else: img_ = np.copy(img)
        fig, ax = plt.subplots(1)
        ax.imshow(img_)
        xmin, ymin, xmax, ymax = bbox
        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        return fig

    @staticmethod
    def crop_w_bbox(img, bbox, offset):
        im = img.copy()
        xmin, ymin, xmax, ymax = bbox
        return im[int(np.floor(ymin-offset)):int(np.ceil(ymax+offset)), int(np.floor(xmin-offset)):int(np.ceil(xmax+offset))]

    @staticmethod
    def translate_landmark(landmark, bbox, offset):
        xmin, ymin, xmax, ymax = bbox
        return landmark - np.array([xmin-offset, ymin-offset])

class KaggleDataset(Dataset):
    """
    https://www.kaggle.com/drgilermo/face-images-with-marked-landmark-points#__sid=js0
    """
    def __init__(self, path, img_extension="png"):
        super().__init__(path, img_extension)

    def load_landmarks(self, min_index=0, max_index=10000, mask_missing_points=False):
        df = pd.read_csv(os.path.join(self.path, "facial_keypoints.csv"))
        cols = df.columns
        lm = df[min_index:max_index].apply(lambda row: np.array([[row[x], row[y]] for x, y in zip(cols[::2], cols[1::2])]), axis=1).to_numpy()
        if mask_missing_points:
            mask_all_lm =  ~df.isnull().sum(axis=1).astype(bool)
            return np.rollaxis(np.dstack(lm[mask_all_lm[min_index:max_index]]), -1), mask_all_lm.to_numpy() # Stacking, and returning as Nx15x2
        return lm

    def load_images(self, min_index=0, max_index=10000):
        img = np.load(os.path.join(self.path, "face_images.npz"))['face_images'][:, :, min_index:max_index]
        return img

    def load(self, min_index=0, max_index=10000, mask_missing_points=True, train_test_split=True):
        if mask_missing_points:
            Y, mask = self.load_landmarks(min_index, max_index, mask_missing_points)
            X =  self.load_images(min_index, max_index)[:, :, mask[min_index:max_index]]
            X = np.rollaxis(X, -1) # Reshapinig at N x W x H
        else:
            X, Y = self.load_images(min_index, max_index), self.load_landmarks(min_index, max_index)
            X = np.rollaxis(X, -1)

        if train_test_split:
            train_indices, val_indices, test_indices = self.read_test_train_split()
            train_indices, val_indices, test_indices = list(map(lambda l : list(filter(lambda x: (x < min(max_index, X.shape[0])) and (x >= min_index), l)), [train_indices, val_indices, test_indices]))
            X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
            Y_train, Y_val, Y_test = Y[train_indices], Y[val_indices], Y[test_indices]

            return X_train, X_val, X_test, Y_train, Y_val, Y_test

        else:
            return X, Y

    def create_train_test_split(self):
        # Get number of indexes
        _, mask = self.load_landmarks(mask_missing_points=True)
        len_ = sum(mask)
        indices = np.arange(len_)
        np.random.shuffle(indices)
        train_indices, val_indices, test_indices = np.split(indices, [int(len_*.6), int(len_*.8)])

        with open(self.path + "/indices.txt", "w") as f:
            train_str = "train_index:" + ",".join([str(x) for x in sorted(train_indices)]) + "\n"
            val_str = "val_index:" + ",".join([str(x) for x in sorted(val_indices)]) + "\n"
            test_str = "test_index:" + ",".join([str(x) for x in sorted(test_indices)]) + "\n"
            out_str = train_str + val_str + test_str
            f.write(out_str)

    @staticmethod
    def matplotlib_visualize_landmark(img, landmark):
        img_ = np.copy(img)

        fig, ax = plt.subplots(1)
        ax.imshow(img_, cmap="gray")
        ax.scatter([x[0] for x in landmark], [x[1] for x in landmark], c="b")
        return fig

    @staticmethod
    def matplotlib_visualize_landmarks(img, landmarks, lm_names):
        img_ = np.copy(img)
        fig, ax = plt.subplots(1)
        ax.imshow(img_, cmap="gray")
        for landmark, lm_name in zip(landmarks, lm_names):
            ax.scatter([x[0] for x in landmark], [x[1] for x in landmark], label=lm_name)
        plt.legend()
        return fig

if __name__ == "__main__":

    path = os.path.join("data", "300W", "01_Indoor")
    dataset = iBug300WDataset(path)
    #dataset.create_train_test_split()
    #_, img, landmark = dataset[66]
    #fig = dataset.matplotlib_visualize_landmark(img, landmark)
    # bbox = dataset.create_bbox(landmark)
    # fig = dataset.matplotlib_visualize_bbox(img, bbox)
    # plt.show()
    # X_train, X_val, X_test, Y_train, Y_val, Y_test = dataset.load()
    X_train, X_val, X_test, Y_train, Y_val, Y_test = dataset.load_cropped_resized()
    img_croppped = X_train[1]
    landmark_cropped = Y_train[1]
    fig = dataset.matplotlib_visualize_landmark(img_croppped, landmark_cropped)
    plt.show()

    # path = os.path.join("data", "Kaggle")
    # dataset = KaggleDataset(path)
    # X_train, X_val, X_test, Y_train, Y_val, Y_test = dataset.load(max_index=100)
    # dataset.matplotlib_visualize_landmark(X_train[0], Y_train[0])



