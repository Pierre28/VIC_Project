import matplotlib.pyplot as plt
from utils import pointwise_euclidean_distance
import numpy as np


def draw_mode_variation_single_plot(img, lms, variations, mode, tag=""):
    fig = plt.figure()
    plt.subplot(1, 1, 1)
    plt.imshow(img, cmap="gray")
    for i, (lm, var) in enumerate(zip(lms, variations)):
        plt.scatter(lm[:, 0], lm[:, 1], label="{:d}".format(var))  # , c="black", alpha=float((i+1)/len(lms)))
    # plt.legend()
    plt.xticks([])
    plt.yticks([])
    # plt.title("Mode : {}".format(mode))
    plt.savefig("fig/{}_iteralgo_{}.png".format(tag, mode), bbox_inches='tight')
    plt.show()
    return fig


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


def display_candidate(cand):
    plt.imshow(cand, cmap="gray")
    plt.show()