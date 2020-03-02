import os
import matplotlib.pyplot as plt
import numpy as np

import argparse

from dataset import KaggleDataset, iBug300WDataset
from dataset_utils import KaggleDatasetUtils
from utils import pointwise_euclidean_distance
from model import ASM

MODELS = {"ASM": ASM}

DATASETS = {"Kaggle": KaggleDataset,
            "iBug": iBug300WDataset}

# TODO : Resize iBug images to same resolution ?

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Experiment main")

    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    model_name = args.model
    dataset_name = args.dataset


    # Import data
    print("Loading data..")
    data_path = os.path.join("data", "Kaggle")
    dataset = DATASETS[dataset_name](data_path)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = dataset.load(min_index=0, max_index=100)

    # Build stat model
    dataset_processor = KaggleDatasetUtils(X_train, Y_train)
    lm_stat_model = dataset_processor.create_lm_stat_model(k=5, strategy=None)
    sm_means, sm_inv_covs = dataset_processor.format_stat_model(lm_stat_model)

    # Train model
    print("Training model..")
    model = MODELS[model_name]
    model.fit(X_train, Y_train)
    model.sm_means, model.sm_inv_covs = sm_means, sm_inv_covs

    # Predict & visualize
    print("Evaluating..")
    lm = model.predict(X_val)
    dataset.matplotlib_visualize_landmark(X_val[0], lm[0])
    plt.show()

    # Evaluate
    dist = pointwise_euclidean_distance(Y_val, lm)
    print(f"PED of model {model_name} on {dataset_name} : {dist}")


