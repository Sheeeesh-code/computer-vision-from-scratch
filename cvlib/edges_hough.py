"""Canny edge detection and Hough transforms for lines and circles.

Coordinate convention: for an image ``img[x, y]``, ``x`` is the row index and
``y`` the column index (origin at the top-left corner).
"""

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from skimage.draw import circle_perimeter
from skimage.feature import peak_local_max


# --------------------------------------------------------------------------
# Canny edge detector
# --------------------------------------------------------------------------

def make_gaussian_kernel(ksize, sigma):
    """Normalised 2-D Gaussian kernel of shape (ksize, ksize).

    The centre sits at ``ksize // 2``, so even sizes use the same centre as the
    next larger odd size.
    """
    center = ksize // 2
    x, y = np.meshgrid(np.arange(ksize), np.arange(ksize))
    kernel = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def filter_image(image, kernel):
    """2-D convolution (reflect border) of a grayscale image with an odd kernel."""
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("kernel dimensions must be odd")
    padded = np.pad(image, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode="symmetric")
    windows = sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel[::-1, ::-1])


def estimate_gradients(img, scale=255.0):
    """Sobel gradient magnitude and orientation (radians).

    ``x`` points down (rows) and ``y`` points right (columns). The image is
    divided by ``scale`` first.
    """
    img = img / scale
    kx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    ky = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64)
    dx = filter_image(img, kx)
    dy = filter_image(img, ky)
    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dy, dx)
    angle[magnitude < 1e-12] = 0.0
    return magnitude, angle


def non_maximum_suppression(d_mag, d_angle):
    """Thin edges: keep a pixel only if it is >= its two neighbours along the
    gradient direction. The direction is quantised into 4 sectors (no
    interpolation)."""
    deg = np.degrees(d_angle)[1:-1, 1:-1]
    c = d_mag[1:-1, 1:-1]
    up, down = d_mag[:-2, 1:-1], d_mag[2:, 1:-1]
    left, right = d_mag[1:-1, :-2], d_mag[1:-1, 2:]
    up_left, down_right = d_mag[:-2, :-2], d_mag[2:, 2:]
    up_right, down_left = d_mag[:-2, 2:], d_mag[2:, :-2]

    vertical = ((deg >= -22.5) & (deg < 22.5)) | (deg >= 157.5) | (deg < -157.5)
    diag1 = ((deg >= 22.5) & (deg < 67.5)) | ((deg >= -157.5) & (deg < -112.5))
    horizontal = ((deg >= 67.5) & (deg < 112.5)) | ((deg >= -112.5) & (deg < -67.5))

    n1 = np.select([vertical, diag1, horizontal], [up, up_left, left], up_right)
    n2 = np.select([vertical, diag1, horizontal], [down, down_right, right], down_left)

    out = np.zeros_like(d_mag)
    out[1:-1, 1:-1] = np.where((c >= n1) & (c >= n2), c, 0)
    return out


def _bilinear(arr, r, c):
    """Bilinear sample of ``arr`` at float coordinates (0 outside the image)."""
    h, w = arr.shape
    r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
    r1, c1 = r0 + 1, c0 + 1
    valid = (r0 >= 0) & (c0 >= 0) & (r1 < h) & (c1 < w)
    r0, r1 = np.clip(r0, 0, h - 1), np.clip(r1, 0, h - 1)
    c0, c1 = np.clip(c0, 0, w - 1), np.clip(c1, 0, w - 1)
    wr, wc = r - np.floor(r), c - np.floor(c)
    top = (1 - wc) * arr[r0, c0] + wc * arr[r0, c1]
    bottom = (1 - wc) * arr[r1, c0] + wc * arr[r1, c1]
    return np.where(valid, (1 - wr) * top + wr * bottom, 0.0)


def non_maximum_suppression_interpol(d_mag, d_angle):
    """Same as :func:`non_maximum_suppression`, but the two neighbours are
    bilinearly interpolated exactly along the gradient direction."""
    h, w = d_mag.shape
    rows, cols = np.mgrid[1:h - 1, 1:w - 1]
    dr, dc = np.cos(d_angle[1:-1, 1:-1]), np.sin(d_angle[1:-1, 1:-1])
    c = d_mag[1:-1, 1:-1]
    p = _bilinear(d_mag, rows + dr, cols + dc)
    n = _bilinear(d_mag, rows - dr, cols - dc)
    out = np.zeros_like(d_mag)
    out[1:-1, 1:-1] = np.where((c >= p) & (c >= n), c, 0)
    return out


def double_thresholding(inp, perc_weak=0.1, perc_strong=0.3):
    """Split thinned edges into binary (weak, strong) maps using thresholds
    placed at a fraction of the [min, max] range of ``inp``."""
    lo, hi = inp.min(), inp.max()
    high_t = lo + perc_strong * (hi - lo)
    low_t = lo + perc_weak * (hi - lo)
    strong = inp >= high_t
    weak = (inp >= low_t) & (inp < high_t)
    return weak, strong


def edge_linking(weak, strong, n=200):
    """Hysteresis: promote weak pixels touching (8-neighbourhood) a strong
    pixel, repeating until nothing changes or ``n`` passes are done."""
    assert weak.shape == strong.shape, "weak and strong maps must have the same shape"
    strong = strong.copy()
    weak = weak.copy()
    for _ in range(n):
        m = np.pad(strong, 1, mode="constant", constant_values=False)
        near_strong = (m[:-2, :-2] | m[:-2, 1:-1] | m[:-2, 2:] | m[1:-1, :-2]
                       | m[1:-1, 2:] | m[2:, :-2] | m[2:, 1:-1] | m[2:, 2:])
        promoted = weak & near_strong
        if not promoted.any():
            break
        strong |= promoted
        weak &= ~promoted
    return strong


def canny(img, ksize=5, sigma=1.0, perc_weak=0.1, perc_strong=0.3,
          interpolate=False, return_gradients=False):
    """Full Canny pipeline: blur -> gradients -> NMS -> double threshold -> linking.

    Returns the boolean edge map, plus ``(magnitude, angle)`` if
    ``return_gradients`` is true.
    """
    blurred = filter_image(img.astype(np.float64), make_gaussian_kernel(ksize, sigma))
    magnitude, angle = estimate_gradients(blurred)
    nms = non_maximum_suppression_interpol if interpolate else non_maximum_suppression
    weak, strong = double_thresholding(nms(magnitude, angle), perc_weak, perc_strong)
    edges = edge_linking(weak, strong)
    return (edges, magnitude, angle) if return_gradients else edges


# --------------------------------------------------------------------------
# Hough transform: lines
# --------------------------------------------------------------------------

def hough_vote_lines(edges, d_rho=1.0, d_theta=np.pi / 180):
    """Vote for lines ``rho = x cos(theta) + y sin(theta)``.

    Returns ``(A, rhos, thetas)`` where ``A[r, t]`` counts the edge pixels
    voting for ``(rhos[r], thetas[t])``.
    """
    h, w = edges.shape
    diag = np.sqrt(h ** 2 + w ** 2)
    thetas = np.arange(0, np.pi, d_theta)
    rhos = np.arange(-diag, diag, d_rho)

    xs, ys = np.nonzero(edges)
    rho_per_vote = xs[:, None] * np.cos(thetas) + ys[:, None] * np.sin(thetas)
    rho_idx = ((rho_per_vote + diag) / d_rho).astype(np.int64)
    theta_idx = np.broadcast_to(np.arange(len(thetas)), rho_idx.shape)

    valid = (rho_idx >= 0) & (rho_idx < len(rhos))
    flat = rho_idx[valid] * len(thetas) + theta_idx[valid]
    A = np.bincount(flat, minlength=len(rhos) * len(thetas))
    return A.reshape(len(rhos), len(thetas)).astype(np.int32), rhos, thetas


def find_peak_params(hspace, params_list, window_size=1, threshold=0.5):
    """Local maxima of a Hough space.

    Keeps bins above ``threshold * max`` that are maximal within
    ``min_distance=window_size``. Returns ``[values, params_0, params_1, ...]``
    where ``params_k`` holds the parameter value of dimension ``k`` for each
    peak.
    """
    assert hspace.ndim == len(params_list), "one parameter array per dimension"
    for i, p in enumerate(params_list):
        assert hspace.shape[i] == len(p), f"dimension {i}: {len(p)} values vs {hspace.shape[i]} bins"
    peaks = peak_local_max(hspace.copy(), exclude_border=False,
                           threshold_rel=threshold, min_distance=window_size)
    values = np.array([hspace[tuple(p)] for p in peaks])
    return [values] + [params_list[i][peaks.T[i]] for i in range(len(params_list))]


# --------------------------------------------------------------------------
# Hough transform: circles
# --------------------------------------------------------------------------

def _radius_range(shape, radius):
    h, w = shape[:2]
    if radius is None:
        return 3, int(np.hypot(h, w))
    return int(radius[0]), int(radius[1])


def _circle_offsets(r):
    """Unique (dx, dy) offsets of a discrete circle of radius ``r``.

    ``circle_perimeter`` repeats a few points (octant symmetry); each cell must
    receive a single vote per edge pixel.
    """
    dx, dy = circle_perimeter(0, 0, int(r))
    return np.unique(np.stack([dx, dy], axis=1), axis=0).T


def hough_vote_circles(edges, radius=None):
    """Circle Hough transform.

    Each edge pixel votes for every circle of radius ``r`` passing through it,
    i.e. for a pre-computed discrete circle of centres around itself.

    Returns ``(A, R, X, Y)`` with ``A`` of shape (len(R), H, W) indexed by
    (radius, row, col).
    """
    h, w = edges.shape[:2]
    r_min, r_max = _radius_range(edges.shape, radius)
    R, X, Y = np.arange(r_min, r_max + 1), np.arange(h), np.arange(w)
    A = np.zeros((len(R), h, w), dtype=np.int32)
    xs, ys = np.nonzero(edges)

    for k, r in enumerate(R):
        dx, dy = _circle_offsets(r)
        cx = (xs[:, None] + dx[None, :]).ravel()
        cy = (ys[:, None] + dy[None, :]).ravel()
        inside = (cx >= 0) & (cx < h) & (cy >= 0) & (cy < w)
        A[k] = np.bincount(cx[inside] * w + cy[inside], minlength=h * w).reshape(h, w)
    return A, R, X, Y


def hough_vote_circles_weighted(edges, radius=None):
    """Circle Hough transform with size-normalised votes.

    A circle of radius ``r`` collects roughly ``r`` times more votes than a
    small one, so each vote is weighted by ``1 / sqrt(perimeter points)``; every
    radius slice is lightly blurred, and the accumulator is square-rooted to
    flatten the remaining dynamic range.
    """
    h, w = edges.shape[:2]
    r_min, r_max = _radius_range(edges.shape, radius)
    R, X, Y = np.arange(r_min, r_max + 1), np.arange(h), np.arange(w)
    A = np.zeros((len(R), h, w), dtype=np.float64)
    xs, ys = np.nonzero(edges)

    for k, r in enumerate(R):
        dx, dy = circle_perimeter(0, 0, int(r))
        if len(dx) == 0:
            continue
        cx = (xs[:, None] + dx[None, :]).ravel()
        cy = (ys[:, None] + dy[None, :]).ravel()
        inside = (cx >= 0) & (cx < h) & (cy >= 0) & (cy < w)
        counts = np.bincount(cx[inside] * w + cy[inside], minlength=h * w).reshape(h, w)
        A[k] = cv2.GaussianBlur(counts / np.sqrt(len(dx)), (3, 3), 0)
    return np.sqrt(np.maximum(A, 0.0)), R, X, Y


def hough_vote_circles_grad(edges, d_angle, radius=None, step=1):
    """Gradient-based circle Hough transform.

    Instead of a full circle of votes, each edge pixel casts only two votes per
    radius, along its gradient direction (towards and away from the centre).
    ``step`` is the bin size, in pixels, for radius and centre coordinates.
    """
    h, w = edges.shape[:2]
    r_min, r_max = _radius_range(edges.shape, radius)
    R = np.arange(r_min, r_max + 1, step)
    X, Y = np.arange(0, h, step), np.arange(0, w, step)
    A = np.zeros((len(R), len(X), len(Y)), dtype=np.int32)

    xs, ys = np.nonzero(edges)
    cos_a, sin_a = np.cos(d_angle[xs, ys]), np.sin(d_angle[xs, ys])

    for k, r in enumerate(R):
        for sign in (1, -1):
            cx = np.round((xs + sign * r * cos_a) / step).astype(int)
            cy = np.round((ys + sign * r * sin_a) / step).astype(int)
            ok = (cx >= 0) & (cx < len(X)) & (cy >= 0) & (cy < len(Y))
            np.add.at(A[k], (cx[ok], cy[ok]), 1)
    return A, R, X, Y
