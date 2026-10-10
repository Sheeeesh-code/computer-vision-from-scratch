"""Sanity tests: agreement with OpenCV and with analytic / synthetic cases."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvlib import edges_hough as eh
from cvlib import features as ft
from cvlib import segmentation as seg
from cvlib import template_matching as tm


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (40, 50), dtype=np.uint8)
    return img, img[8:17, 12:22].copy()


# ---------------------------------------------------------------- template matching

def test_ncc_variants_agree_gray(scene):
    img, template = scene
    ref = tm.ncc_naive(img, template)
    assert np.allclose(ref, tm.ncc_fast(img, template))
    assert np.allclose(ref, tm.ncc_matrix(img, template))


def test_ncc_variants_agree_colour():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (25, 30, 3), dtype=np.uint8)
    template = img[5:11, 7:13].copy()
    ref = tm.ncc_naive(img, template)
    assert np.allclose(ref, tm.ncc_fast(img, template))
    assert np.allclose(ref, tm.ncc_matrix(img, template))


def test_ncc_matches_opencv(scene):
    img, template = scene
    assert np.allclose(tm.ncc_matrix(img, template),
                       cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED), atol=1e-4)
    assert np.allclose(tm.ncc_mean_subtracted(img, template),
                       cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED), atol=1e-4)


def test_ncc_finds_template(scene):
    img, template = scene
    response = tm.ncc_mean_subtracted(img, template)
    assert np.unravel_index(response.argmax(), response.shape) == (8, 12)
    assert response.max() == pytest.approx(1.0)


def test_mean_subtraction_is_gain_offset_invariant(scene):
    img, template = scene
    changed = 0.5 * img.astype(float) + 40
    response = tm.ncc_mean_subtracted(changed, template)
    assert np.unravel_index(response.argmax(), response.shape) == (8, 12)
    assert response.max() == pytest.approx(1.0)


def test_pad_zeros_and_gray():
    out = tm.pad_zeros(np.ones((1, 1), dtype=int), 1, 2, 3, 4)
    assert out.shape == (4, 8) and out.sum() == 1 and out[1, 3] == 1
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[..., 1] = 100
    assert (tm.rgb2gray(rgb) == int(0.587 * 100)).all()


def test_non_max_suppression_keeps_isolated_peaks():
    response = np.zeros((20, 20))
    response[5, 5], response[5, 7], response[15, 15] = 0.9, 0.8, 0.7
    peaks = tm.non_max_suppression(response, (2, 2), threshold=0.5)
    assert set(zip(*peaks.nonzero())) == {(5, 5), (15, 15)}


# ---------------------------------------------------------------- Canny / Hough

def test_gaussian_kernel_normalised():
    kernel = eh.make_gaussian_kernel(7, 1.5)
    assert kernel.sum() == pytest.approx(1.0)
    assert kernel[3, 3] == kernel.max()


def test_canny_detects_square_outline():
    img = np.zeros((60, 60))
    img[15:45, 15:45] = 255
    edges = eh.canny(img, ksize=5, sigma=1.0, perc_weak=0.05, perc_strong=0.3)
    ys, xs = edges.nonzero()
    assert edges.sum() > 80
    assert ys.min() >= 12 and ys.max() <= 47 and xs.min() >= 12 and xs.max() <= 47
    assert not edges[25:35, 25:35].any()  # nothing inside the square


def test_hough_lines_recovers_orientation():
    edges = np.zeros((80, 80), dtype=bool)
    edges[40, 10:70] = True  # horizontal line, x (row) = 40
    A, rhos, thetas = eh.hough_vote_lines(edges)
    r, t = np.unravel_index(A.argmax(), A.shape)
    assert A.max() == 60
    assert thetas[t] == pytest.approx(0.0, abs=1e-9)
    assert rhos[r] == pytest.approx(40, abs=1.0)


def test_hough_circles_recovers_centre_and_radius():
    from skimage.draw import circle_perimeter
    edges = np.zeros((100, 100), dtype=bool)
    rr, cc = circle_perimeter(45, 55, 20)
    edges[rr, cc] = True
    for fn in (eh.hough_vote_circles, eh.hough_vote_circles_weighted):
        A, R, X, Y = fn(edges, radius=(10, 30))
        k, x, y = np.unravel_index(A.argmax(), A.shape)
        assert (R[k], X[x], Y[y]) == (20, 45, 55)


def test_gradient_hough_circles():
    from skimage.draw import disk
    img = np.zeros((100, 100))
    rr, cc = disk((45, 55), 20)
    img[rr, cc] = 255
    edges, _, angle = eh.canny(img, ksize=5, sigma=1.0, perc_weak=0.05,
                               perc_strong=0.3, return_gradients=True)
    A, R, X, Y = eh.hough_vote_circles_grad(edges, angle, radius=(10, 30))
    k, x, y = np.unravel_index(A.argmax(), A.shape)
    assert abs(R[k] - 20) <= 1 and abs(X[x] - 45) <= 1 and abs(Y[y] - 55) <= 1


# ---------------------------------------------------------------- segmentation

def test_mean_shift_two_blobs():
    rng = np.random.default_rng(0)
    points = np.vstack([rng.normal(0, 1, (40, 2)), rng.normal(20, 1, (40, 2))])
    for fn in (seg.mean_shift_segmentation, seg.mean_shift_segmentation_fast):
        labels = fn(points, bandwidth=4.0)
        assert labels.max() + 1 == 2
        assert len(set(labels[:40])) == 1 and len(set(labels[40:])) == 1


def test_filter_kernels():
    gabor = seg.create_gabor_filter(sigma=4, theta=0, lambd=8)
    assert gabor.shape == (25, 25) and gabor[12, 12] == pytest.approx(1.0)
    log = seg.create_log_filter(2)
    assert log.shape == (13, 13) and log.sum() == pytest.approx(0.0, abs=1e-9)
    assert len(seg.default_filter_bank()) == 12


def test_texton_histogram_matches_bruteforce():
    rng = np.random.default_rng(0)
    texton_map = rng.integers(0, 6, (15, 17))
    hist = seg.compute_texton_histogram(texton_map, 6, 5)
    assert np.allclose(hist.sum(axis=2), 1.0)
    for i, j in [(0, 0), (7, 8), (14, 16)]:
        window = texton_map[max(0, i - 2):i + 3, max(0, j - 2):j + 3]
        expected = np.bincount(window.ravel(), minlength=6) / window.size
        assert np.allclose(hist[i, j], expected)


def test_clustering_accuracy_is_permutation_invariant():
    truth = np.array([[0, 0, 1, 1], [2, 2, 1, 1]])
    assert seg.clustering_accuracy((truth + 1) % 3, truth) == 1.0


# ---------------------------------------------------------------- features

def test_homography_recovers_known_transform():
    rng = np.random.default_rng(1)
    h = np.array([[1.1, 0.05, 12], [-0.03, 0.95, -7], [1e-4, 2e-4, 1]])
    src = rng.uniform(0, 300, (30, 2))
    dst = ft.transform_homography(src, h)
    assert np.allclose(ft.compute_homography(src, dst), h, atol=1e-8)


def test_ransac_rejects_outliers():
    rng = np.random.default_rng(2)
    h = np.array([[1.0, 0.1, 5], [0.0, 1.0, -3], [0, 0, 1]])
    pts1 = rng.uniform(0, 200, (60, 2))
    pts2 = ft.transform_homography(pts1, h)
    pts2[:15] = rng.uniform(0, 200, (15, 2))  # 25% wrong matches
    matches = np.stack([np.arange(60), np.arange(60)], axis=1)
    est, inliers = ft.ransac_homography(pts1, pts2, matches, n_samples=6,
                                        n_iters=300, delta=1.0, seed=0)
    assert np.allclose(est, h, atol=1e-6)
    assert len(inliers) >= 44


def test_mirror_descriptor_is_an_involution():
    desc = np.random.default_rng(0).random(128)
    assert np.allclose(ft.shift_sift_descriptor(ft.shift_sift_descriptor(desc)), desc)


def test_simple_sift_is_unit_norm_and_ratio_test():
    rng = np.random.default_rng(0)
    patches = [rng.random((16, 16)) for _ in range(6)]
    descs = np.array([ft.simple_sift(p) for p in patches])
    assert descs.shape == (6, 128) and np.allclose(np.linalg.norm(descs, axis=1), 1.0)
    matches = ft.ratio_test_match(descs, descs, 0.8)
    assert (matches[:, 0] == matches[:, 1]).all() and len(matches) == 6


def test_harris_finds_square_corners():
    img = np.zeros((60, 60), dtype=np.float32)
    img[20:40, 20:40] = 1.0
    corners = ft.harris_keypoints(img, min_distance=5, threshold_rel=0.2, margin=2)
    assert len(corners) >= 4
    for target in [(20, 20), (20, 39), (39, 20), (39, 39)]:
        assert np.min(np.linalg.norm(corners - target, axis=1)) <= 3
