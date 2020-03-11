import os
import matplotlib.pyplot as plt
import numpy as np

import argparse
from tqdm import tqdm

from dataset import KaggleDataset, iBug300WDataset
from dataset_utils import KaggleDatasetUtils, iBug300DatasetUtils
from utils import pointwise_euclidean_distance
from model import ASM

MODELS = {"ASM": ASM}

DATASETS = {"Kaggle": KaggleDataset,
            "300W": iBug300WDataset}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Experiment main")

    parser.add_argument("--model", type=str, default="ASM", choices=["ASM"])
    parser.add_argument("--dataset", type=str, default="300W", choices=["Kaggle", "300W"])
    parser.add_argument("--strategy", type=str, default=None, help="preprocessing applied before computing stat model",
                        choices=["laplacian", "all_dir", None])
    parser.add_argument("--distance", type=str, default="mah", help="fit function", choices=["mah", "mse"])
    parser.add_argument("--descriptor", type=str, default="sift", help="Type of descriptor used to compute point stat model",
                        choices=["sift", None])
    parser.add_argument("--k", type=int, default=20, help="Size of the neigborhood used to compute point statistical model")
    parser.add_argument("--m", type=int, default=50, help="Range of the search when finding best candidates around point")

    args = parser.parse_args()

    model_name = args.model
    dataset_name = args.dataset
    strategy = args.strategy
    dist = args.distance
    descriptor = args.descriptor
    k = args.k
    m = args.m

    assert bool(descriptor) != bool(strategy), "Strategy and descriptor cannot be use simutaneously"


    # Import data
    print("Loading data..")
    data_path = os.path.join("data", "300W", "01_Indoor") if dataset_name == "300W" else os.path.join("data", "Kaggle")
    dataset = DATASETS[dataset_name](data_path)
    Loader = dataset.load_cropped_resized if dataset_name == "300W" else dataset.load
    X_train, X_val, X_test, Y_train, Y_val, Y_test = Loader()

    # Build stat model
    DatasetProcessor = iBug300DatasetUtils if dataset_name == "300W" else KaggleDatasetUtils
    dataset_processor = DatasetProcessor(X_train, Y_train)
    lm_stat_model = dataset_processor.create_lm_stat_model(k=k, strategy=strategy, descriptor=descriptor)
    sm_means, sm_inv_covs = dataset_processor.format_stat_model(lm_stat_model)

    # Train model
    print("Training model..")
    model = MODELS[model_name]()
    model.fit(X_train, Y_train)
    model.sm_means, model.sm_inv_covs = sm_means, sm_inv_covs



    # Predict & visualize
    print("Evaluating..")
    X_val_filtered = dataset_processor.transform_img_with_respect_to_strat(X_val, strategy=strategy)[:10]
    pred_lm = []
    for img in tqdm(X_val_filtered):
        X_ = model.asm(dataset_processor, img, m, max_it=5, dist=dist, descriptor=descriptor, patch_size=2*k+1)
        pred_lm.append(X_)
    pred_lm = np.array(pred_lm)


    lm = model.predict(X_val)
    # for id in range(10, 30):
        # dataset.matplotlib_visualize_landmarks(X_val[id], [pred_lm[id], model.mean_.reshape((-1, 2)), Y_val[id]],
        #                               ["pred", "mean", "gt"])
        # plt.show()

    # Evaluate
    dist = pointwise_euclidean_distance(Y_val[:10], pred_lm[:10])
    dist_mean = pointwise_euclidean_distance(Y_val[:10], lm[:10])
    print(f"PED of model {model_name} on {dataset_name} : {dist} (Mean model : {dist_mean}")



    # Temporary
    import cv2
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints = sift.detect(np.uint8(X_train[0]), None)
    img = cv2.drawKeypoints(np.uint8(X_train[0]), keypoints, np.uint8(X_train[0]))
    cv2.imshow("img", img)
    kp, des = sift.detectAndCompute(np.uint8(X_train[0]), None)

    def draw_mode_variation_single_plot(img, lms, variations, mode, tag=""):
        fig = plt.figure()
        plt.subplot(1, 1, 1)
        plt.imshow(img, cmap="gray")
        for i, (lm, var) in enumerate(zip(lms, variations)):
            plt.scatter(lm[:, 0], lm[:, 1], label="{:.1f}".format(var), c="black", alpha=float((i+1)/len(lms)))
        # plt.legend()s
        plt.xticks([])
        plt.yticks([])
        # plt.title("Mode : {}".format(mode))
        plt.savefig("fig/{}_mode_{}.png".format(tag, mode), bbox_inches='tight')
        plt.show()
        return fig


    from matplotlib import cm
    import matplotlib as mpl
    #mode = 3 # mode 3 nez bouche mode 4 ouverture bouche mode 5 yeux
    for mode in range(0, 10):
        lms, variations = model.create_mode_map(mode, 5)
        img = np.ones_like(X_train[5])*255
        img[0, 0] = 0
        draw_mode_variation_single_plot(img, lms, variations, mode, tag="300W")

    fig = dataset.matplotlib_visualize_landmark(X_train[0], Y_train[0])
    for i in range(6, 9):
        pass
        dataset.matplotlib_visualize_landmark(X_train[i], Y_train[i])
        plt.show()



    lapl_mse = [2.9, 2.62, 2.4, 2.6, 2.5]
    sift_mse = [2.6, 2.2, 2.2, 2.19, 2.4]
    all_mse = [2.87, 2.6, 2.4, 2.43]
    none_mse = [5.4, 5.3, 4.4, 4.8, 5.2]
    sift_mah = [2.5, 1.39, 1.31, 1.45, 1.57]

    k = [5, 10, 12, 15, 30]
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(k, lapl_mse, label="Laplacian - MSE", alpha=.6, lw=.6, marker="+")
    plt.plot(k, sift_mse, label="SIFT - MSE", alpha=.6, lw=.6, marker="+")
    # plt.plot(k, all_mse, label="Laplacian - MSE", alpha=.6, lw=.6, marker="+")
    plt.plot(k, none_mse, label="Raw - MSE", alpha=.6, lw=.6, marker="+")
    plt.plot(k, sift_mah, label="SIFT - Mahalanobis", alpha=.6, lw=.6, marker="+")
    plt.legend()
    plt.xlabel("Descriptor patch size")
    plt.ylabel("Pointwise Euclidean Distance")
    plt.savefig("fig/plot.png",  bbox_inches='tight')
    plt.show()

    def display_lm(img, lm, lm_gt=None, legend=False, id=0, tag=""):
        plt.imshow(img, cmap="gray")
        plt.scatter(lm[:, 0], lm[:,1], c="r", label="pred")
        if lm_gt is not None :
            plt.scatter(lm_gt[:, 0], lm_gt[:, 1], c="g", label="gt")
            plt.title("PED : {}".format(pointwise_euclidean_distance(lm, lm_gt)))
        if legend: plt.legend()
        plt.xticks([])
        plt.yticks([])
        plt.savefig("fig/img_{}_{}_{:.3f}.png".format(tag, id, pointwise_euclidean_distance(lm, lm_gt)), bbox_inches='tight')
        plt.show()

    for id in range(0, 49):
        display_lm(X_val[id], pred_lm[id], Y_val[id], legend=True, id=id)
        plt.show()
