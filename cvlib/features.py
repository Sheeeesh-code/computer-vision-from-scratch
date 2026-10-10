"""Keypoints, descriptors, matching and two applications: image stitching
(homography + RANSAC) and mirror-symmetry detection (mirrored SIFT descriptors
+ Hough voting).

Coordinate convention: keypoints are ``(row, col)`` = ``(y, x)`` pairs, as in
``img[y, x]``.
"""

import cv2
import numpy as np
from scipy.ndimage import convolve
from scipy.spatial.distance import cdist
from skimage import filters
from skimage.feature import corner_peaks

from .edges_hough import find_peak_params


# --------------------------------------------------------------------------
# Keypoint detection
# --------------------------------------------------------------------------

def harris_corners(img, window_size=3, k=0.04):
    """Harris corner response ``R = det(M) - k * trace(M)^2``.

    The second-moment matrix ``M = [[A, B], [B, C]]`` sums ``Ix^2``, ``IxIy``
    and ``Iy^2`` over a uniform ``window_size`` square (box filter by
    convolution).
    """
    ix = filters.sobel_v(img)
    iy = filters.sobel_h(img)
    window = np.ones((window_size, window_size))
    a = convolve(ix * ix, window)
    b = convolve(ix * iy, window)
    c = convolve(iy * iy, window)
    return (a * c - b ** 2) - k * (a + c) ** 2


def filter_border_keypoints(keypoints, shape, margin):
    """Keep keypoints at least ``margin`` pixels away from every image border."""
    keypoints = np.asarray(keypoints)
    ok = ((keypoints[:, 0] >= margin) & (keypoints[:, 0] < shape[0] - margin)
          & (keypoints[:, 1] >= margin) & (keypoints[:, 1] < shape[1] - margin))
    return keypoints[ok]


def harris_keypoints(img, min_distance=8, threshold_rel=0.01, margin=8, **harris_kwargs):
    """Harris corners as an (N, 2) array of ``(row, col)``, with local-maximum
    selection and keypoints closer than ``margin`` pixels to the border dropped
    (so that a full descriptor patch always fits)."""
    response = harris_corners(img, **harris_kwargs)
    keypoints = corner_peaks(response, min_distance=min_distance,
                             threshold_rel=threshold_rel, exclude_border=False)
    return filter_border_keypoints(keypoints, img.shape, margin)


# --------------------------------------------------------------------------
# Descriptors
# --------------------------------------------------------------------------

def naive_descriptor(patch):
    """Flattened patch, normalised to zero mean and unit variance."""
    normalised = (patch - patch.mean()) / (patch.std() + 1e-4)
    return normalised.flatten().astype(np.float32)


def _gaussian_window(size, sigma):
    ax = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
    yy, xx = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (yy ** 2 + xx ** 2) / sigma ** 2)
    return kernel / kernel.sum()


def simple_sift(patch, sigma=3.0):
    """Simplified SIFT descriptor (128-D, not rotation invariant).

    The 16x16 patch is split into a 4x4 grid of cells; each cell gets an
    8-bin histogram of gradient orientations, voting with gradient magnitude
    times a Gaussian window centred on the patch. The concatenation is
    normalised to unit length.
    """
    size = patch.shape[0]
    assert patch.shape == (size, size) and size % 4 == 0, "patch must be square, size divisible by 4"
    cell = size // 4

    ix = filters.sobel_v(patch)
    iy = filters.sobel_h(patch)
    weight = np.hypot(ix, iy) * _gaussian_window(size, sigma)
    orientation = np.degrees(np.arctan2(iy, ix)) % 360
    bins = (orientation // 45).astype(int) % 8

    def to_cells(a):  # (size, size) -> (16 cells, cell*cell pixels)
        return a.reshape(4, cell, 4, cell).transpose(0, 2, 1, 3).reshape(16, -1)

    weight_c, bins_c = to_cells(weight), to_cells(bins)
    hist = np.stack([np.bincount(bins_c[i], weights=weight_c[i], minlength=8)
                     for i in range(16)])
    feature = hist.ravel()
    norm = np.linalg.norm(feature)
    return feature / norm if norm > 0 else feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """Apply ``desc_func`` to the square patch around each ``(row, col)`` keypoint."""
    half_before, half_after = patch_size // 2, (patch_size + 1) // 2
    descriptors = []
    for y, x in keypoints:
        patch = image[max(0, y - half_before):y + half_after,
                      max(0, x - half_before):x + half_after]
        descriptors.append(desc_func(patch))
    return np.array(descriptors)


def compute_cv2_descriptor(img, method=None):
    """Keypoints and descriptors from OpenCV (SIFT by default).

    Returns ``(keypoints, descriptors, angles, sizes)`` with keypoints as
    ``(row, col)``.
    """
    method = method or cv2.SIFT_create()
    kpts, descs = method.detectAndCompute(img, None)
    keypoints = np.array([(kp.pt[1], kp.pt[0]) for kp in kpts])
    angles = np.array([kp.angle for kp in kpts])
    sizes = np.array([kp.size for kp in kpts])
    return keypoints, descs, angles, sizes


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

def top_k_matches(desc1, desc2, k=2):
    """For each descriptor of ``desc1``, its ``k`` nearest descriptors in
    ``desc2``: ``[(i, [(j, distance), ...]), ...]``."""
    distances = cdist(desc1, desc2, metric="euclidean")
    matches = []
    for i in range(len(desc1)):
        nearest = np.argsort(distances[i])[:k]
        matches.append((i, [(int(j), float(distances[i, j])) for j in nearest]))
    return matches


def ratio_test_match(desc1, desc2, match_threshold):
    """Lowe's ratio test: keep ``i -> a`` if ``d(i, a) / d(i, b) < threshold``,
    where ``a`` and ``b`` are the two nearest neighbours. Returns an (M, 2)
    array of ``[index in desc1, index in desc2]``."""
    pairs = []
    for i, neighbours in top_k_matches(desc1, desc2, k=2):
        if len(neighbours) < 2:
            continue
        (best, d_best), (_, d_second) = neighbours
        if d_best < match_threshold * d_second:
            pairs.append([i, best])
    return np.array(pairs, dtype=int).reshape(-1, 2)


# --------------------------------------------------------------------------
# Homography and RANSAC
# --------------------------------------------------------------------------

def transform_homography(src, h_matrix, normalize=True):
    """Apply a 3x3 homography to (N, 2) points."""
    homogeneous = np.insert(np.asarray(src, dtype=np.float64), 2, 1.0, axis=1)
    out = h_matrix @ homogeneous.T
    if normalize:
        out = out[:-1] / out[-1]
    return out.T


def _normalization_matrix(points):
    """Similarity transform: zero centroid, mean distance sqrt(2)."""
    mean = points.mean(axis=0)
    scale = np.sqrt(2) / np.mean(np.linalg.norm(points - mean, axis=1))
    return np.array([[scale, 0, -scale * mean[0]],
                     [0, scale, -scale * mean[1]],
                     [0, 0, 1]])


def compute_homography(src, dst):
    """Homography ``H`` with ``dst ~ H @ src`` from >= 4 correspondences, by the
    normalised Direct Linear Transform (null-space of the DLT system via SVD)."""
    t_src, t_dst = _normalization_matrix(src), _normalization_matrix(dst)
    src_n = transform_homography(src, t_src)
    dst_n = transform_homography(dst, t_dst)

    rows = []
    for (x, y), (xp, yp) in zip(src_n, dst_n):
        rows.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        rows.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
    _, _, vh = np.linalg.svd(np.array(rows))
    h = np.linalg.inv(t_dst) @ vh[-1].reshape(3, 3) @ t_src
    return h / h[2, 2]


def ransac_homography(keypoints1, keypoints2, matches, sampling_ratio=0.5,
                      n_iters=500, delta=20, min_samples=4, n_samples=None, seed=None):
    """Robust homography from ``keypoints1`` to ``keypoints2`` with RANSAC.

    Each iteration fits a homography on a random subset of the matches
    (``n_samples``, or ``sampling_ratio`` of them), counts inliers (projection
    within ``delta`` pixels) and keeps the largest inlier set. The final
    homography is re-estimated on all inliers.

    Returns ``(H, inlier_matches)``.
    """
    rng = np.random.default_rng(seed)
    pts1 = keypoints1[matches[:, 0]]
    pts2 = keypoints2[matches[:, 1]]
    n = len(matches)
    if n_samples is None:
        n_samples = int(n * sampling_ratio)
    n_samples = min(max(n_samples, min_samples), n)

    best = np.zeros(n, dtype=bool)
    for _ in range(n_iters):
        idx = rng.choice(n, n_samples, replace=False)
        h = compute_homography(pts1[idx], pts2[idx])
        error = np.linalg.norm(transform_homography(pts1, h) - pts2, axis=1)
        inliers = error < delta
        if inliers.sum() > best.sum():
            best = inliers

    if best.sum() < min_samples:
        raise RuntimeError("RANSAC found fewer than 4 inliers; loosen delta or add matches")
    return compute_homography(pts1[best], pts2[best]), matches[best]


def stitch_images(src, dst, h_matrix):
    """Warp ``dst`` into the frame of ``src`` and paste ``src`` on top.

    ``h_matrix`` maps ``(row, col)`` points of ``src`` to those of ``dst``.
    The canvas has the height of ``src`` and the combined width of both images.
    """
    # The homography is expressed in (row, col); OpenCV expects (x, y),
    # hence the transposes.
    src_t, dst_t = cv2.transpose(src).copy(), cv2.transpose(dst).copy()
    canvas = cv2.warpPerspective(dst_t, np.linalg.inv(h_matrix),
                                 (src_t.shape[1], src_t.shape[0] + dst_t.shape[0]))
    canvas[:src_t.shape[0], :src_t.shape[1]] = src_t
    return cv2.transpose(canvas)


def trim(frame):
    """Remove fully black border rows and columns."""
    mask = frame.reshape(frame.shape[0], frame.shape[1], -1).sum(axis=2) > 0
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if len(rows) == 0:
        return frame
    return frame[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


# --------------------------------------------------------------------------
# Mirror symmetry
# --------------------------------------------------------------------------

def angle_with_x_axis(pi, pj):
    """Angle in [0, pi) of the line through two ``(row, col)`` points, w.r.t. the x (column) axis."""
    dy, dx = pi[0] - pj[0], pi[1] - pj[1]
    if dx == 0:
        return np.pi / 2
    angle = np.arctan(dy / dx)
    return angle + np.pi if angle < 0 else angle


def midpoint(pi, pj):
    """Midpoint of two ``(row, col)`` points."""
    return (pi[0] + pj[0]) / 2, (pi[1] + pj[1]) / 2


def shift_sift_descriptor(desc):
    """Descriptor of the horizontally mirrored keypoint, obtained directly from
    the 128-D SIFT vector: the 4x4 cell grid is flipped and, inside each cell,
    the orientation histogram is reversed (bin 0 stays in place, since SIFT
    already puts the dominant orientation first)."""
    grid = desc.reshape(4, 4, 8)[::-1, :, :]
    grid = grid[:, :, [0, 7, 6, 5, 4, 3, 2, 1]]
    return grid.reshape(-1)


def create_mirror_descriptors(img):
    """SIFT keypoints and descriptors of an image, plus the virtual descriptors
    of its mirror image (same keypoint order)."""
    kps, descs, angles, sizes = compute_cv2_descriptor(img)
    mirrored = np.array([shift_sift_descriptor(d) for d in descs])
    return kps, descs, angles, sizes, mirrored


def match_mirror_descriptors(descs, mirror_descs, threshold=0.7):
    """Match each keypoint with the mirror descriptor of *another* keypoint.

    The 3 nearest mirror descriptors are found, the one coming from the same
    keypoint is dropped, and the ratio test is applied to the best two left.
    """
    result = []
    for idx, neighbours in top_k_matches(descs, mirror_descs, k=3):
        kept = [(j, d) for j, d in neighbours if j != idx][:2]
        if len(kept) < 2:
            kept = neighbours[:2]
        if len(kept) < 2:
            continue
        (best, d_best), (_, d_second) = kept
        if d_second <= 1e-12:
            ratio = 0.0 if d_best <= 1e-12 else np.inf
        else:
            ratio = d_best / d_second
        if ratio < threshold:
            result.append([idx, best])
    return np.asarray(result, dtype=int).reshape(-1, 2)


def find_symmetry_lines(matches, kps):
    """One candidate symmetry line per matched pair ``(i, j)``.

    The segment ``i-j`` is perpendicular to the mirror line and the mirror line
    goes through its midpoint, so the line is ``rho = x cos(theta) + y sin(theta)``
    with ``theta`` the angle of the segment and ``(x, y)`` = (column, row) the
    midpoint. A vertical mirror axis at column ``c`` is ``theta = 0, rho = c``.
    """
    rhos, thetas = [], []
    for i, j in matches:
        pi, pj = kps[i], kps[j]
        theta = angle_with_x_axis(pi, pj)
        y_mid, x_mid = midpoint(pi, pj)
        rhos.append(x_mid * np.cos(theta) + y_mid * np.sin(theta))
        thetas.append(theta)
    return rhos, thetas


def hough_vote_mirror(matches, kps, im_shape, window=1, threshold=0.5, num_lines=1):
    """Hough voting for the mirror line over ``(rho, theta)``.

    Returns the ``num_lines`` strongest ``(rhos, thetas)``.
    """
    rhos, thetas = find_symmetry_lines(matches, kps)
    diagonal = int(np.ceil(np.hypot(im_shape[0], im_shape[1])))
    rho_values = np.arange(-diagonal, diagonal + 1, 1)
    theta_values = np.deg2rad(np.arange(0, 360, 1))

    hspace = np.zeros((len(rho_values), len(theta_values)))
    for rho, theta in zip(rhos, thetas):
        rho_idx = np.argmin(np.abs(rho_values - rho))
        theta_idx = np.argmin(np.abs(theta_values - theta % (2 * np.pi)))
        hspace[rho_idx, theta_idx] += 1

    values, peak_rhos, peak_thetas = find_peak_params(
        hspace, [rho_values, theta_values], window_size=window, threshold=threshold)
    if len(values) == 0:
        return np.array([]), np.array([])
    order = np.argsort(values)[::-1][:num_lines]
    return peak_rhos[order], peak_thetas[order]
