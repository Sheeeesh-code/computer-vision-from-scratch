"""Segmentation demo: K-means vs Mean-Shift on colour, then texture
segmentation with a Gabor/LoG filter bank and textons.

Run from the repository root:  python examples/03_segmentation.py
Writes figures to results/.
"""

import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

from cvlib import segmentation as seg
from cvlib import viz


def colour_demo():
    full = data.astronaut()
    image = cv2.resize(full, (48, 48), interpolation=cv2.INTER_AREA)
    features = seg.construct_feature_space(image, spatial_weight=1.0)

    bandwidths = (15, 30, 60)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Input (48x48)")
    axes[1, 0].imshow(full)
    axes[1, 0].set_title("Original (512x512)")
    rows = ["| method | parameter | segments | time (s) |", "|---|---|---|---|"]

    for col, bw in enumerate(bandwidths, start=1):
        start = time.perf_counter()
        labels = seg.mean_shift_segmentation_fast(features, bw).reshape(image.shape[:2])
        elapsed = time.perf_counter() - start
        n = labels.max() + 1
        viz.show_segmentation(labels, axes[0, col], f"Mean-Shift, bandwidth {bw}: {n} segments")

        start = time.perf_counter()
        km = seg.kmeans_segment(image, n_clusters=max(n, 2))
        km_elapsed = time.perf_counter() - start
        viz.show_segmentation(km, axes[1, col], f"K-means, k={max(n, 2)}")
        rows.append(f"| Mean-Shift (fast) | bandwidth {bw} | {n} | {elapsed:.2f} |")
        rows.append(f"| K-means | k = {max(n, 2)} | {max(n, 2)} | {km_elapsed:.2f} |")

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Colour segmentation: the Mean-Shift bandwidth sets the number of segments, K-means needs k")
    fig.tight_layout()
    fig.savefig(RESULTS / "segmentation_meanshift_kmeans.png", dpi=90)
    plt.close(fig)
    (RESULTS / "segmentation_meanshift.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


def filter_bank_demo():
    bank = seg.default_filter_bank()
    names = [f"Gabor sigma={s} theta={t}" for s in (2, 4) for t in (0, 45, 90, 135)]
    names += [f"LoG sigma={s}" for s in (1, 2, 4, 8)]

    fig, axes = plt.subplots(2, 6, figsize=(18, 6.5))
    for ax, kernel, name in zip(axes.ravel(), bank, names):
        ax.imshow(kernel, cmap="RdBu_r")
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle("Filter bank: 8 Gabor filters (2 scales x 4 orientations) + 4 Laplacian-of-Gaussian")
    fig.tight_layout()
    fig.savefig(RESULTS / "filter_bank_kernels.png", dpi=90)
    plt.close(fig)

    image = data.astronaut()[:256, 128:384]
    responses = seg.build_filter_bank_responses(image, bank)
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.5))
    for k, (ax, name) in enumerate(zip(axes.ravel(), names)):
        ax.imshow(responses[..., k], cmap="gray")
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle("Filter-bank responses: each orientation highlights edges perpendicular to it")
    fig.tight_layout()
    fig.savefig(RESULTS / "filter_bank_responses.png", dpi=90)
    plt.close(fig)


def make_mosaic(size=192):
    textures = {"brick": data.brick(), "grass": data.grass(), "gravel": data.gravel()}
    mosaic = np.zeros((size, size), np.uint8)
    truth = np.zeros((size, size), int)
    half = size // 2
    mosaic[:, :half] = textures["brick"][:size, :half]
    mosaic[:half, half:] = textures["grass"][:half, :half]
    truth[:half, half:] = 1
    mosaic[half:, half:] = textures["gravel"][:half, :half]
    truth[half:, half:] = 2
    # Training patches come from a different part of each texture.
    train = [np.stack([t[256:384, s:s + 128]] * 3, -1) for t in textures.values() for s in (0, 128)]
    return np.stack([mosaic] * 3, -1), truth, train


def texture_demo():
    image, truth, train = make_mosaic()
    bank = seg.default_filter_bank()

    start = time.perf_counter()
    dictionary = seg.TextonDictionary(bank, n_textons=32).learn_dictionary(train)
    learn_time = time.perf_counter() - start

    results = {}
    results["Gray-level K-means"] = seg.kmeans_segment(image, 3)
    results["Smoothed filter responses"] = seg.segment_smoothed_responses(image, bank, 3)
    for window in (15, 51):
        labels, texton_map = seg.segment_texton_histograms(image, dictionary, 3, window_size=window)
        results[f"Texton histograms (window {window})"] = labels

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Texture mosaic (brick | grass / gravel)")
    viz.show_segmentation(truth, axes[0, 1], "Ground truth")
    viz.show_segmentation(texton_map, axes[0, 2], "Texton map (32 textons)", cmap="nipy_spectral")
    axes[0, 3].axis("off")
    rows = ["| method | pixel accuracy |", "|---|---|"]
    for ax, (name, labels) in zip(axes[1], results.items()):
        acc = seg.clustering_accuracy(labels, truth)
        viz.show_segmentation(labels, ax, f"{name}\naccuracy {acc:.2f}")
        rows.append(f"| {name} | {acc:.2f} |")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "segmentation_textures.png", dpi=90)
    plt.close(fig)
    (RESULTS / "segmentation_textures.md").write_text(
        "Texture mosaic (skimage.data brick / grass / gravel), 3 clusters; "
        f"texton dictionary learned in {learn_time:.1f}s\n\n" + "\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    colour_demo()
    filter_bank_demo()
    texture_demo()
