"""Plotting helpers used by the example scripts."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def show_images(images, titles=None, figsize=None, cmap_gray=True):
    """Display images side by side. Returns the figure."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize or (4 * n, 4), squeeze=False)
    for i, (ax, im) in enumerate(zip(axes[0], images)):
        ax.imshow(im, cmap="gray" if (cmap_gray and im.ndim == 2) else None)
        ax.axis("off")
        if titles is not None:
            ax.set_title(titles[i])
    fig.tight_layout()
    return fig


def draw_lines(img, hspace, rhos, thetas, peaks, figsize=(15, 5)):
    """Input, Hough accumulator and detected lines.

    ``peaks`` is the output of ``find_peak_params`` on the accumulator. Lines
    are ``rho = x cos(theta) + y sin(theta)`` with x the row index.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    d_theta, d_rho = np.diff(thetas).mean() / 2, np.diff(rhos).mean() / 2
    extent = [np.rad2deg(thetas[0] - d_theta), np.rad2deg(thetas[-1] + d_theta),
              rhos[-1] + d_rho, rhos[0] - d_rho]
    axes[1].imshow(np.log1p(hspace), extent=extent, cmap="gray", aspect=1 / 1.5)
    axes[1].set_title("Hough accumulator")
    axes[1].set_xlabel("theta (deg)")
    axes[1].set_ylabel("rho (px)")

    axes[2].imshow(img, cmap="gray")
    axes[2].set_title("Detected lines")
    axes[2].axis("off")
    for _, rho, theta in zip(*peaks):
        # Point of the line closest to the origin, in (col, row) plot coordinates.
        p0 = np.array([rho * np.sin(theta), rho * np.cos(theta)])
        direction = np.array([np.cos(theta), -np.sin(theta)])
        axes[2].axline(tuple(p0), tuple(p0 + direction), color="tab:red", linewidth=1.2)
    axes[2].set_xlim(0, img.shape[1])
    axes[2].set_ylim(img.shape[0], 0)
    fig.tight_layout()
    return fig


def draw_circles(img, peaks, title="Detected circles", ax=None):
    """Overlay circles ``peaks = [values, radii, rows, cols]`` on an image."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    for _, r, x, y in zip(*peaks):
        ax.add_patch(Circle((y, x), r, color="red", fill=False, linewidth=1.5))
    ax.set_title(title)
    ax.axis("off")
    return ax


def plot_matches(ax, image1, image2, keypoints1, keypoints2, matches,
                 keypoints_color="k", matches_color=None, only_matches=False, seed=0):
    """Draw two images side by side with lines between matched keypoints.

    Keypoints are ``(row, col)`` arrays; ``matches`` is an (Q, 2) index array.
    """
    rng = np.random.default_rng(seed)
    height = max(image1.shape[0], image2.shape[0])

    def pad_to(im):
        out = np.zeros((height,) + im.shape[1:], dtype=im.dtype)
        out[:im.shape[0]] = im
        return out

    image1, image2 = pad_to(image1), pad_to(image2)
    offset = image1.shape[1]
    ax.imshow(np.concatenate([image1, image2], axis=1), interpolation="nearest", cmap="gray")

    if not only_matches:
        ax.scatter(keypoints1[:, 1], keypoints1[:, 0], facecolors="none", edgecolors=keypoints_color)
        ax.scatter(keypoints2[:, 1] + offset, keypoints2[:, 0], facecolors="none", edgecolors=keypoints_color)

    for i, j in matches:
        color = rng.random(3) if matches_color is None else matches_color
        ax.plot((keypoints1[i, 1], keypoints2[j, 1] + offset),
                (keypoints1[i, 0], keypoints2[j, 0]), "-", color=color, linewidth=0.8)
    ax.axis("off")


def draw_symmetry_lines(img, rhos, thetas, matches=None, kps=None, ax=None, title="Mirror line"):
    """Overlay mirror lines ``rho = x cos(theta) + y sin(theta)`` with x the
    column and y the row (the convention of ``features.find_symmetry_lines``)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    if matches is not None and kps is not None:
        for i, j in matches:
            ax.plot([kps[i, 1], kps[j, 1]], [kps[i, 0], kps[j, 0]],
                    color="yellow", linewidth=0.5, alpha=0.6)
    for rho, theta in zip(rhos, thetas):
        p0 = np.array([rho * np.cos(theta), rho * np.sin(theta)])  # (col, row)
        direction = np.array([-np.sin(theta), np.cos(theta)])
        ax.axline(tuple(p0), tuple(p0 + direction), color="red", linestyle="--", linewidth=2)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_title(title)
    ax.axis("off")
    return ax


def show_segmentation(labels, ax=None, title=None, cmap="tab20"):
    """Display a label map."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(labels, cmap=cmap, interpolation="nearest")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax
