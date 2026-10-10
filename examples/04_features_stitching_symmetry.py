"""Keypoints, matching, image stitching (RANSAC) and mirror-symmetry detection.

Run from the repository root:  python examples/04_features_stitching_symmetry.py
Writes figures to results/.
"""

import sys
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

from cvlib import features as ft
from cvlib import viz


def to_gray_float(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def make_pair(img, overlap=0.4, shear=2.5e-4, angle=0.0):
    """Split an image into two overlapping halves; the right one is then seen
    from a slightly different viewpoint (perspective warp, optional rotation)."""
    h, w = img.shape[:2]
    cut = int(w * (0.5 + overlap / 2))
    start = int(w * (0.5 - overlap / 2))
    left = img[:, :cut].copy()
    right = img[:, start:].copy()
    rh, rw = right.shape[:2]
    rotation = cv2.getRotationMatrix2D((rw / 2, rh / 2), angle, 1.0)
    homography = np.vstack([rotation, [0, 0, 1]]) @ np.array([[1, 0, 0], [0, 1, 0], [shear, shear, 1]])
    right = cv2.warpPerspective(right, homography, (rw, rh))
    return left, right


def matching_demo():
    img = data.rocket()
    left, right = make_pair(img)
    g1, g2 = to_gray_float(left), to_gray_float(right)

    k1 = ft.harris_keypoints(g1, min_distance=8, threshold_rel=0.01)
    k2 = ft.harris_keypoints(g2, min_distance=8, threshold_rel=0.01)

    fig, axes = plt.subplots(2, 1, figsize=(13, 11))
    summary = ["| descriptor | ratio threshold | matches |", "|---|---|---|"]
    for ax, (name, fn, size) in zip(axes, (("naive (normalised patch)", ft.naive_descriptor, 8),
                                           ("simplified SIFT", ft.simple_sift, 16))):
        d1 = ft.describe_keypoints(g1, k1, fn, size)
        d2 = ft.describe_keypoints(g2, k2, fn, size)
        for thr in (0.6, 0.8, 0.95):
            summary.append(f"| {name} | {thr} | {len(ft.ratio_test_match(d1, d2, thr))} |")
        matches = ft.ratio_test_match(d1, d2, 0.8)
        viz.plot_matches(ax, left, right, k1, k2, matches, only_matches=True)
        ax.set_title(f"{name}: {len(matches)} matches (ratio test 0.8)")
    fig.tight_layout()
    fig.savefig(RESULTS / "feature_matching.png", dpi=80)
    plt.close(fig)
    (RESULTS / "feature_matching.md").write_text(
        f"Harris keypoints ({len(k1)} / {len(k2)}) on two views of skimage.data.rocket\n\n"
        + "\n".join(summary) + "\n")
    print("\n".join(summary))


def stitching_demo(name, image, overlap, shear, angle, rotation_invariant=False, seed=0):
    """``rotation_invariant`` swaps the simplified SIFT (Harris keypoints, fixed
    orientation) for OpenCV's SIFT, which is needed as soon as the views are rotated."""
    left, right = make_pair(image, overlap, shear, angle)
    if rotation_invariant:
        k1, d1, _, _ = ft.compute_cv2_descriptor(cv2.cvtColor(left, cv2.COLOR_RGB2GRAY))
        k2, d2, _, _ = ft.compute_cv2_descriptor(cv2.cvtColor(right, cv2.COLOR_RGB2GRAY))
    else:
        g1, g2 = to_gray_float(left), to_gray_float(right)
        k1 = ft.harris_keypoints(g1, min_distance=8, threshold_rel=0.005)
        k2 = ft.harris_keypoints(g2, min_distance=8, threshold_rel=0.005)
        d1 = ft.describe_keypoints(g1, k1, ft.simple_sift)
        d2 = ft.describe_keypoints(g2, k2, ft.simple_sift)
    matches = ft.ratio_test_match(d1, d2, 0.8)

    h_plain = ft.compute_homography(k1[matches[:, 0]], k2[matches[:, 1]])
    h_ransac, inliers = ft.ransac_homography(k1, k2, matches, n_samples=8,
                                             n_iters=2000, delta=3, seed=seed)

    plain = ft.trim(ft.stitch_images(left, right, h_plain))
    robust = ft.trim(ft.stitch_images(left, right, h_ransac))

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
    ax = fig.add_subplot(grid[0, 0])
    viz.plot_matches(ax, left, right, k1, k2, matches, only_matches=True)
    ax.set_title(f"{len(matches)} putative matches")
    ax = fig.add_subplot(grid[0, 1])
    viz.plot_matches(ax, left, right, k1, k2, inliers, only_matches=True)
    ax.set_title(f"{len(inliers)} RANSAC inliers")
    ax = fig.add_subplot(grid[1, 0])
    ax.imshow(plain)
    ax.set_title("Stitched with all matches (least squares)")
    ax.axis("off")
    ax = fig.add_subplot(grid[1, 1])
    ax.imshow(robust)
    ax.set_title("Stitched with RANSAC")
    ax.axis("off")
    fig.suptitle(f"Image stitching: {name} ("
                 + ("OpenCV SIFT, rotation invariant" if rotation_invariant else "simplified SIFT") + ")")
    fig.tight_layout()
    fig.savefig(RESULTS / f"stitching_{name}.png", dpi=75)
    plt.close(fig)
    print(f"stitching {name}: {len(matches)} matches, {len(inliers)} inliers")
    return len(matches), len(inliers)


def symmetry_demo():
    """Mirror line of a symmetric image, found by matching SIFT descriptors
    against the descriptors of the mirrored image."""
    half = data.astronaut()[40:360, 150:310]
    gray = cv2.cvtColor(np.concatenate([half, half[:, ::-1]], axis=1), cv2.COLOR_RGB2GRAY)
    image = np.concatenate([half, half[:, ::-1]], axis=1)

    kps, descs, _, _, mirrored = ft.create_mirror_descriptors(gray)
    matches = ft.match_mirror_descriptors(descs, mirrored, threshold=0.7)
    rhos, thetas = ft.hough_vote_mirror(matches, kps, gray.shape, window=10,
                                        threshold=0.5, num_lines=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    axes[0].imshow(image)
    axes[0].set_title("Input (left half mirrored on the right)")
    axes[0].axis("off")
    viz.draw_symmetry_lines(image, rhos, thetas, matches, kps, ax=axes[1],
                            title=f"{len(matches)} mirror matches, detected axis")
    fig.tight_layout()
    fig.savefig(RESULTS / "mirror_symmetry.png", dpi=90)
    plt.close(fig)

    expected_col = half.shape[1] - 0.5
    if len(rhos):
        # Vertical axis: theta = 0 and rho = column of the axis.
        print(f"symmetry: {len(matches)} matches; axis theta={np.rad2deg(thetas[0]):.0f} deg, "
              f"rho={rhos[0]:.1f} (true column {expected_col:.1f})")
    return rhos, thetas, expected_col


if __name__ == "__main__":
    matching_demo()
    stitching_demo("rocket", data.rocket(), overlap=0.4, shear=2.5e-4, angle=0.0)
    stitching_demo("rocket_rotated", data.rocket(), overlap=0.4, shear=2.5e-4, angle=20.0, rotation_invariant=True)
    symmetry_demo()
