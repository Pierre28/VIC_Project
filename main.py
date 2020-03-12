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
    parser.add_argument("--k", type=int, default=30, help="Size of the neigborhood used to compute point statistical model")
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



    # Predict
    print("Evaluating..")
    num_img = 20
    X_val_filtered = dataset_processor.transform_img_with_respect_to_strat(X_val, strategy=strategy)[:num_img]
    pred_lm = []
    for img in tqdm(X_val_filtered):
        X_ = model.asm(dataset_processor, img, m, max_it=10, dist=dist, descriptor=descriptor, patch_size=2*k+1)
        pred_lm.append(X_)
    pred_lm = np.array(pred_lm)
    lm = model.predict(X_val)


    # Evaluate
    dist = pointwise_euclidean_distance(Y_val[:num_img], pred_lm[:num_img])
    dist_mean = pointwise_euclidean_distance(Y_val[:num_img], lm[:num_img])
    print(f"PED of model {model_name} on {dataset_name} : {dist} (Mean model : {dist_mean}")


