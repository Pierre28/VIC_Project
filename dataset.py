import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

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


class iBug300WDataset(Dataset):

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

    def load(self):
        X, Y = self.load_images(), self.load_landmarks()
        return X, Y


    def load_cropped(self):
        images = []
        landmarks = []
        for index in self.indexes:
            im = self.load_image(index, convert_color=True)
            lm = self.load_landmark(index)
            bbox = self.create_bbox(lm)
            images.append(self.crop_w_bbox(im, bbox))
            landmarks.append(self.translate_landmark(lm, bbox))
        return np.array(images), np.array(landmarks)

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

    @staticmethod
    def matplotlib_visualize_landmark(img, landmark, size=5, is_rgb=True):
        if not is_rgb: img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else: img_ = np.copy(img)
        for i, (x, y) in enumerate(landmark):
            x, y = int(x), int(y)
            img_[y-size:y+size, x-size:x+size, :] = [255, 255, 255]

        fig, ax = plt.subplots(1)
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
    def crop_w_bbox(img, bbox):
        im = img.copy()
        xmin, ymin, xmax, ymax = bbox
        return im[int(np.floor(ymin)):int(np.ceil(ymax)), int(np.floor(xmin)):int(np.ceil(xmax))]

    @staticmethod
    def translate_landmark(landmark, bbox):
        xmin, ymin, xmax, ymax = bbox
        return landmark - np.array([xmin, ymin])

class KaggleDataset(Dataset):

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

    def load(self, min_index=0, max_index=10000, mask_missing_points=True):
        if mask_missing_points:
            Y, mask = self.load_landmarks(min_index, max_index, mask_missing_points)
            X =  self.load_images(min_index, max_index)[:, :, mask[min_index:max_index]]
            X = np.rollaxis(X, -1) # Reshapinig at N x W x H
            return X, Y
        else:
            X, Y = self.load_images(min_index, max_index), self.load_landmarks(min_index, max_index)
            X = np.rollaxis(X, -1)
            return X, Y

    @staticmethod
    def matplotlib_visualize_landmark(img, landmark):
        img_ = np.copy(img)

        fig, ax = plt.subplots(1)
        ax.imshow(img_, cmap="gray")
        ax.scatter([x[0] for x in landmark], [x[1] for x in landmark], c="b")
        return fig


if __name__ == "__main__":

    # path = os.path.join("data", "300W", "01_Indoor")
    # dataset = iBug300WDataset(path)
    # _, img, landmark = dataset[66]
    # fig = dataset.matplotlib_visualize_landmark(img, landmark)
    # bbox = dataset.create_bbox(landmark)
    # fig = dataset.matplotlib_visualize_bbox(img, bbox)
    # plt.show()
    # X, Y = dataset.load()
    # X_aligned, Y_aligned = dataset.load_cropped()
    # img_croppped = dataset.crop_w_bbox(img, bbox)
    # landmark_cropped = dataset.translate_landmark(landmark, bbox)

    path = os.path.join("data", "Kaggle")
    dataset = KaggleDataset(path)
    X, Y = dataset.load()
    dataset.matplotlib_visualize_landmark(X[0], Y[0])



