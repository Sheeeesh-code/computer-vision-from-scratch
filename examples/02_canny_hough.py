"""Canny edge detection and Hough transforms (lines, circles) demo.

Run from the repository root:  python examples/02_canny_hough.py
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

from cvlib import edges_hough as eh
from cvlib import viz


def edge_agreement(ours, reference, tolerance=1):
    """Precision / recall of ``ours`` against ``reference`` with a pixel tolerance."""
    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    near_ref = cv2.dilate(reference.astype(np.uint8), kernel) > 0
    near_ours = cv2.dilate(ours.astype(np.uint8), kernel) > 0
    precision = (ours & near_ref).sum() / max(ours.sum(), 1)
    recall = (reference & near_ours).sum() / max(reference.sum(), 1)
    return precision, recall


def canny_demo():
    img = data.camera().astype(np.float64)
    blurred = eh.filter_image(img, eh.make_gaussian_kernel(5, 1.0))
    magnitude, angle = eh.estimate_gradients(blurred)
    nms = eh.non_maximum_suppression(magnitude, angle)
    nms_interp = eh.non_maximum_suppression_interpol(magnitude, angle)
    weak, strong = eh.double_thresholding(nms, perc_weak=0.03, perc_strong=0.12)
    edges = eh.edge_linking(weak, strong)

    reference = cv2.Canny(data.camera(), 50, 150, L2gradient=True) > 0
    precision, recall = edge_agreement(edges, reference)

    panels = [img, magnitude, nms, strong * 1.0 + weak * 0.5, edges, reference]
    titles = ["Input", "Gradient magnitude", "After NMS", "Strong (white) / weak (grey)",
              "After edge linking (ours)", "cv2.Canny (reference)"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, im, title in zip(axes.ravel(), panels, titles):
        ax.imshow(im, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"Canny pipeline from scratch. Agreement with OpenCV (1 px tolerance): "
                 f"precision {precision:.2f}, recall {recall:.2f}")
    fig.tight_layout()
    fig.savefig(RESULTS / "canny_pipeline.png", dpi=90)
    plt.close(fig)

    diff = np.abs(nms - nms_interp)
    print(f"Canny: our edges {edges.sum()} px, OpenCV {reference.sum()} px, "
          f"precision {precision:.2f}, recall {recall:.2f}; "
          f"NMS with/without interpolation differ on {(diff > 0).sum()} px")


def lines_demo():
    img = data.checkerboard()
    h, w = img.shape
    rotation = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0)
    rotated = cv2.warpAffine(img, rotation, (w, h), borderMode=cv2.BORDER_REFLECT)

    edges = eh.canny(rotated.astype(np.float64), ksize=5, sigma=1.5,
                     perc_weak=0.05, perc_strong=0.3)
    accumulator, rhos, thetas = eh.hough_vote_lines(edges)
    peaks = eh.find_peak_params(accumulator, [rhos, thetas], window_size=12, threshold=0.4)

    fig = viz.draw_lines(rotated, accumulator, rhos, thetas, peaks)
    fig.suptitle(f"Hough transform for lines: {len(peaks[0])} peaks (checkerboard rotated by 15 deg)", y=1.02)
    fig.savefig(RESULTS / "hough_lines.png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"Lines: {len(peaks[0])} peaks, theta mod 90 deg ~ "
          f"{np.round(np.median(np.rad2deg(peaks[2]) % 90), 1)}")


def circles_demo():
    gray = data.coins()
    edges, _, angle = eh.canny(gray.astype(np.float64), ksize=7, sigma=2.0,
                               perc_weak=0.05, perc_strong=0.2, return_gradients=True)
    radius = (16, 60)

    methods = [
        ("Unweighted votes", lambda: eh.hough_vote_circles(edges, radius), 0.6),
        ("Weighted votes", lambda: eh.hough_vote_circles_weighted(edges, radius), 0.7),
        ("Gradient-based votes", lambda: eh.hough_vote_circles_grad(edges, angle, radius), 0.4),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
    axes[0].imshow(edges, cmap="gray")
    axes[0].set_title(f"Canny edges ({edges.sum()} px)")
    axes[0].axis("off")

    rows = ["| method | vote time (s) | circles found |", "|---|---|---|"]
    for ax, (name, run, threshold) in zip(axes[1:], methods):
        start = time.perf_counter()
        accumulator, R, X, Y = run()
        elapsed = time.perf_counter() - start
        peaks = eh.find_peak_params(accumulator, [R, X, Y], window_size=15, threshold=threshold)
        viz.draw_circles(gray, peaks, f"{name}: {len(peaks[0])} circles", ax=ax)
        rows.append(f"| {name} | {elapsed:.2f} | {len(peaks[0])} |")
    fig.suptitle("Circle Hough transform on 24 coins, radii 16-60 px")
    fig.tight_layout()
    fig.savefig(RESULTS / "hough_circles.png", dpi=80)
    plt.close(fig)

    (RESULTS / "hough_circles.md").write_text(
        "Circle Hough transform on 24 coins (skimage.data.coins), radii 16-60 px\n\n"
        + "\n".join(rows) + "\n")
    print("\n".join(rows))


def bin_size_demo():
    """Coarser accumulator bins make the gradient-based transform both faster
    and more tolerant to imprecise edge directions."""
    gray = data.coins()
    edges, _, angle = eh.canny(gray.astype(np.float64), ksize=7, sigma=2.0,
                               perc_weak=0.05, perc_strong=0.2, return_gradients=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, step in zip(axes, (1, 3, 5)):
        start = time.perf_counter()
        A, R, X, Y = eh.hough_vote_circles_grad(edges, angle, (16, 60), step=step)
        elapsed = time.perf_counter() - start
        peaks = eh.find_peak_params(A, [R, X, Y], window_size=max(15 // step, 3), threshold=0.4)
        viz.draw_circles(gray, peaks, f"bin size {step}px: {len(peaks[0])} circles ({elapsed:.2f}s)", ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "hough_circles_bin_size.png", dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    canny_demo()
    lines_demo()
    circles_demo()
    bin_size_demo()
