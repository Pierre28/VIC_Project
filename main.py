import os
import matplotlib.pyplot as plt

from dataset import KaggleDataset, iBug300WDataset
from utils import pointwise_euclidean_distance
from model import ASM

# TODO : Resize iBug images to same resolution ?

if __name__ == "__main__":

    # Import data
    data_path = os.path.join("data", "Kaggle")
    dataset = KaggleDataset(data_path)
    X, Y = dataset.load(min_index=0, max_index=100)

    # Train model
    model = ASM()
    model.fit(X, Y)

    # Predict & visualize
    lm = model.predict(X)
    dataset.matplotlib_visualize_landmark(X[0], lm[0])
    plt.show()

    # Evaluate
    dist = pointwise_euclidean_distance(Y, lm)
    print(f"PED of model : {dist}")