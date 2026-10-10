"""Template matching demo: NCC variants, non-maximum suppression, benchmark.

Run from the repository root:  python examples/01_template_matching.py
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

from cvlib import template_matching as tm
from cvlib import viz


def detect(img, template, response_fn, suppress, threshold):
    response = response_fn(img, template)
    peaks = tm.non_max_suppression(response, suppress, threshold)
    # Move peaks from "window top-left" to "window centre" coordinates.
    centred = np.zeros(img.shape[:2])
    h, w = template.shape[:2]
    ys, xs = peaks.nonzero()
    centred[ys + h // 2, xs + w // 2] = peaks[ys, xs]
    return response, centred


def checkerboard_demo():
    """Template taken from the clean image, scene under a different lighting.

    Plain NCC only compares directions of intensity vectors, so a global
    brightness offset makes every window look alike. Subtracting the mean
    removes gain and offset and keeps the response discriminative.
    """
    clean = data.checkerboard().astype(np.float64)
    template = clean[20:55, 20:55]  # one white square plus a 5 px margin

    ramp = np.linspace(0, 40, clean.shape[1])[None, :]
    scenes = {"original lighting": clean,
              "darker + offset (0.4 * I + 100 + ramp)": 0.4 * clean + 100 + ramp}

    fig, axes = plt.subplots(2, 5, figsize=(17, 7))
    counts = {}
    for row, (name, scene) in enumerate(scenes.items()):
        r_plain, p_plain = detect(scene, template, tm.ncc_matrix, (12, 12), 0.95)
        r_ms, p_ms = detect(scene, template, tm.ncc_mean_subtracted, (12, 12), 0.95)
        counts[name] = (int(np.count_nonzero(p_plain)), int(np.count_nonzero(p_ms)))
        axes[row, 0].imshow(scene, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title(name)
        axes[row, 1].imshow(r_plain, cmap="viridis", vmin=-1, vmax=1)
        axes[row, 1].set_title(f"NCC response [{r_plain.min():.2f}, {r_plain.max():.2f}]")
        axes[row, 2].imshow(scene, cmap="gray", vmin=0, vmax=255)
        ys, xs = p_plain.nonzero()
        axes[row, 2].scatter(xs, ys, c="red", s=20)
        axes[row, 2].set_title(f"NCC peaks > 0.95: {len(xs)}")
        axes[row, 3].imshow(r_ms, cmap="viridis", vmin=-1, vmax=1)
        axes[row, 3].set_title(f"Mean-subtracted [{r_ms.min():.2f}, {r_ms.max():.2f}]")
        axes[row, 4].imshow(scene, cmap="gray", vmin=0, vmax=255)
        ys, xs = p_ms.nonzero()
        axes[row, 4].scatter(xs, ys, c="red", s=20)
        axes[row, 4].set_title(f"Mean-subtracted peaks > 0.95: {len(xs)}")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Template: white square (35x35, from the original image), two lighting conditions")
    fig.tight_layout()
    fig.savefig(RESULTS / "template_matching_checkerboard.png", dpi=100)
    plt.close(fig)
    return counts


def coins_demo():
    img = data.coins()
    template = img[104:146, 314:356]  # a small coin

    response, peaks = detect(img, template, tm.ncc_mean_subtracted, (20, 20), 0.55)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(img, cmap="gray")
    axes[0].add_patch(plt.Rectangle((313.5, 103.5), 42, 42, fill=False, ec="red", lw=2))
    axes[0].set_title("Image and template (red)")
    axes[1].imshow(response, cmap="viridis")
    axes[1].set_title("Mean-subtracted NCC response")
    axes[2].imshow(img, cmap="gray")
    ys, xs = peaks.nonzero()
    axes[2].scatter(xs, ys, c="red", s=30)
    axes[2].set_title(f"Peaks > 0.55 ({len(xs)} found)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "template_matching_coins.png", dpi=110)
    plt.close(fig)


def benchmark():
    img = data.coins()[60:150, 20:140]  # 90 x 120
    template = img[20:35, 20:35]  # 15 x 15

    timings = {}
    reference = None
    for name, fn in (("naive (4 loops)", tm.ncc_naive),
                     ("fast (2 loops)", tm.ncc_fast),
                     ("matrix (im2col)", tm.ncc_matrix)):
        repeats = 1 if fn is tm.ncc_naive else 5
        best = np.inf
        for _ in range(repeats):
            start = time.perf_counter()
            out = fn(img, template)
            best = min(best, time.perf_counter() - start)
        timings[name] = best
        if reference is None:
            reference = out
        assert np.allclose(reference, out), name

    best = np.inf
    for _ in range(5):
        start = time.perf_counter()
        cv_out = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
        best = min(best, time.perf_counter() - start)
    timings["OpenCV matchTemplate (C++, FFT)"] = best
    assert np.allclose(reference, cv_out, atol=1e-4)

    slowest = timings["naive (4 loops)"]
    lines = [f"Image {img.shape[1]}x{img.shape[0]}, template {template.shape[1]}x{template.shape[0]} "
             "(grayscale, best of 5 runs; naive run once)", "",
             "| implementation | time (s) | speed-up vs naive |", "|---|---|---|"]
    for name, t in timings.items():
        lines.append(f"| {name} | {t:.4f} | {slowest / t:,.0f}x |")
    (RESULTS / "benchmark_ncc.md").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    names = list(timings)
    ax.barh(names, [timings[n] for n in names], color=["#c44", "#e90", "#4a4", "#58c"])
    ax.set_xscale("log")
    ax.set_xlabel("time (s, log scale)")
    ax.set_title("Normalized cross-correlation: same output, different cost")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(RESULTS / "benchmark_ncc.png", dpi=110)
    plt.close(fig)
    print("\n".join(lines))


def gradients_demo():
    img = data.camera()
    grads = tm.gray2grad(img)
    fig = viz.show_images([np.abs(g) for g in grads],
                          ["horizontal", "vertical", "diagonal 45", "diagonal 135"],
                          figsize=(16, 4.2))
    fig.suptitle("Sobel and Sobel-like gradient magnitudes", y=1.0)
    fig.savefig(RESULTS / "template_matching_gradients.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("checkerboard peaks (plain NCC, mean-subtracted):", checkerboard_demo())
    coins_demo()
    gradients_demo()
    benchmark()
