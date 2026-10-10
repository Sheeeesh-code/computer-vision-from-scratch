"""Image segmentation by clustering, and texture description.

* Colour segmentation: K-means and Mean-Shift on (position, colour) features.
* Texture: Gabor / Laplacian-of-Gaussian filter banks, texton dictionary
  (K-means on filter responses) and per-pixel texton histograms.
"""

import cv2
import numpy as np
from scipy.ndimage import convolve, gaussian_filter
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import KDTree
from sklearn.utils import shuffle


# --------------------------------------------------------------------------
# Colour segmentation
# --------------------------------------------------------------------------

def kmeans_segment(image, n_clusters, n_samples=1000, seed=0):
    """K-means on RGB values. The model is fitted on ``n_samples`` random
    pixels, then every pixel is assigned to its closest centre."""
    h, w, d = image.shape
    pixels = image.reshape(-1, d).astype(np.float64)
    sample = shuffle(pixels, random_state=seed, n_samples=min(n_samples, len(pixels)))
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit(sample)
    return model.predict(pixels).reshape(h, w)


def construct_feature_space(image, spatial_weight):
    """5-D feature per pixel: ``(w*row, w*col, R, G, B)``, shape (H*W, 5)."""
    h, w, _ = image.shape
    rows, cols = np.mgrid[0:h, 0:w]
    spatial = spatial_weight * np.stack([rows.ravel(), cols.ravel()], axis=1)
    return np.concatenate([spatial, image.reshape(-1, 3)], axis=1).astype(np.float64)


def mean_shift_step(data, point, bandwidth):
    """One Mean-Shift update: Gaussian-weighted mean of ``data`` around ``point``."""
    sq_dist = np.sum((data - point) ** 2, axis=1)
    weights = np.exp(-sq_dist / (2 * bandwidth ** 2))
    return weights @ data / weights.sum()


def _merge_modes(modes, bandwidth):
    """Give the same label to all modes closer than ``bandwidth / 2``."""
    labels = np.full(len(modes), -1, dtype=int)
    next_label = 0
    for i in range(len(modes)):
        if labels[i] != -1:
            continue
        close = np.linalg.norm(modes - modes[i], axis=1) < bandwidth / 2
        labels[close & (labels == -1)] = next_label
        next_label += 1
    return labels


def mean_shift_segmentation(data, bandwidth, threshold=1e-3, max_iter=50):
    """Mean-Shift clustering: every point climbs to a density mode; nearby
    modes are then merged. Returns one label per point."""
    modes = np.zeros_like(data)
    for i in range(len(data)):
        point = data[i].copy()
        for _ in range(max_iter):
            new_point = mean_shift_step(data, point, bandwidth)
            shift = np.linalg.norm(new_point - point)
            point = new_point
            if shift < threshold:
                break
        modes[i] = point
    return _merge_modes(modes, bandwidth)


def mean_shift_segmentation_fast(data, bandwidth, threshold=1e-3, max_iter=50):
    """Mean-Shift with the "basin of attraction" speed-up.

    Once a point has converged, every data point within ``bandwidth`` of any
    position on its trajectory is assigned the same mode and skipped later.
    """
    n = len(data)
    done = np.zeros(n, dtype=bool)
    modes = np.zeros_like(data)
    for i in range(n):
        if done[i]:
            continue
        point = data[i].copy()
        path = [point]
        for _ in range(max_iter):
            new_point = mean_shift_step(data, point, bandwidth)
            shift = np.linalg.norm(new_point - point)
            point = new_point
            path.append(point)
            if shift < threshold:
                break
        for p in path:
            close = np.linalg.norm(data - p, axis=1) < bandwidth
            modes[close] = point
            done[close] = True
    return _merge_modes(modes, bandwidth)


# --------------------------------------------------------------------------
# Filter bank
# --------------------------------------------------------------------------

def _kernel_grid(sigma):
    size = int(6 * sigma + 1)
    if size % 2 == 0:
        size += 1
    half = size // 2
    return np.meshgrid(np.arange(-half, half + 1), np.arange(-half, half + 1))


def create_gabor_filter(sigma, theta, lambd, psi=0.0, gamma=0.5):
    """2-D Gabor kernel: a Gaussian envelope (aspect ratio ``gamma``)
    multiplied by a cosine of wavelength ``lambd`` along orientation ``theta``."""
    x, y = _kernel_grid(sigma)
    x_p = x * np.cos(theta) + y * np.sin(theta)
    y_p = -x * np.sin(theta) + y * np.cos(theta)
    envelope = np.exp(-(x_p ** 2 + gamma ** 2 * y_p ** 2) / (2 * sigma ** 2))
    return envelope * np.cos(2 * np.pi * x_p / lambd + psi)


def create_log_filter(sigma):
    """Laplacian-of-Gaussian kernel (zero-mean), a "blob" detector at scale ``sigma``."""
    x, y = _kernel_grid(sigma)
    r2 = x ** 2 + y ** 2
    log = (r2 - 2 * sigma ** 2) / sigma ** 4 * np.exp(-r2 / (2 * sigma ** 2))
    return log - log.mean()


def default_filter_bank():
    """12 filters: Gabor at 4 orientations x 2 scales, plus LoG at 4 scales."""
    thetas = np.deg2rad([0, 45, 90, 135])
    gabors = [create_gabor_filter(s, t, lam) for s, lam in [(2, 4), (4, 8)] for t in thetas]
    logs = [create_log_filter(s) for s in (1, 2, 4, 8)]
    return gabors + logs


def build_filter_bank_responses(image, filter_bank):
    """Convolve the grayscale image with each filter -> (H, W, len(filter_bank))."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    responses = [convolve(gray, kernel, mode="reflect") for kernel in filter_bank]
    return np.stack(responses, axis=-1)


# --------------------------------------------------------------------------
# Textons
# --------------------------------------------------------------------------

class TextonDictionary:
    """Dictionary of textons: cluster centres of filter-bank responses."""

    def __init__(self, filter_bank, n_textons=200, standardize=True):
        self.filter_bank = filter_bank
        self.n_textons = n_textons
        self.standardize = standardize  # per-filter zero mean / unit variance before clustering
        self.mean_ = None
        self.std_ = None
        self.cluster_centers_ = None
        self.tree_ = None

    def _features(self, img):
        responses = build_filter_bank_responses(img, self.filter_bank)
        features = responses.reshape(-1, responses.shape[-1])
        if self.standardize and self.mean_ is not None:
            features = (features - self.mean_) / self.std_
        return responses.shape[:2], features

    def learn_dictionary(self, training_imgs):
        """Cluster the filter responses of all training images (MiniBatchKMeans)."""
        features = np.concatenate([
            build_filter_bank_responses(img, self.filter_bank).reshape(-1, len(self.filter_bank))
            for img in training_imgs
        ])
        if self.standardize:
            self.mean_ = features.mean(axis=0)
            self.std_ = features.std(axis=0) + 1e-12
            features = (features - self.mean_) / self.std_
        kmeans = MiniBatchKMeans(n_clusters=self.n_textons, random_state=0, batch_size=1000)
        kmeans.fit(features)
        self.cluster_centers_ = kmeans.cluster_centers_
        self.tree_ = KDTree(self.cluster_centers_)
        return self

    def assign_textons(self, img):
        """Texton ID (nearest cluster centre) of every pixel -> (H, W) int map."""
        (h, w), features = self._features(img)
        _, indices = self.tree_.query(features, k=1)
        return indices.reshape(h, w)


def compute_texton_histogram(texton_map, n_textons, window_size):
    """Normalised histogram of texton IDs in a ``window_size`` square around
    every pixel -> (H, W, n_textons).

    The window is clipped at the borders (only valid pixels are counted).
    Counts come from an integral image per texton, so the cost does not
    depend on the window size.
    """
    h, w = texton_map.shape
    half = window_size // 2

    one_hot = np.zeros((h, w, n_textons), dtype=np.float32)
    one_hot[np.arange(h)[:, None], np.arange(w)[None, :], texton_map] = 1.0
    integral = np.zeros((h + 1, w + 1, n_textons), dtype=np.float64)
    integral[1:, 1:] = one_hot.cumsum(axis=0).cumsum(axis=1)

    rows = np.arange(h)
    cols = np.arange(w)
    r0, r1 = np.clip(rows - half, 0, h), np.clip(rows + half + 1, 0, h)
    c0, c1 = np.clip(cols - half, 0, w), np.clip(cols + half + 1, 0, w)
    counts = (integral[r1][:, c1] - integral[r0][:, c1]
              - integral[r1][:, c0] + integral[r0][:, c0])
    return counts / counts.sum(axis=2, keepdims=True)


# --------------------------------------------------------------------------
# Texture-based segmentation
# --------------------------------------------------------------------------

def segment_smoothed_responses(image, filter_bank, n_clusters, blur_sigma=12.0,
                               rectify=True, standardize=True, seed=0):
    """K-means on spatially smoothed filter-bank responses.

    ``rectify`` takes absolute values before smoothing (signed responses
    average out to ~0, so only their energy carries texture information) and
    ``standardize`` gives every channel zero mean and unit variance so that no
    filter dominates the clustering.
    """
    responses = build_filter_bank_responses(image, filter_bank)
    if rectify:
        responses = np.abs(responses)
    smoothed = np.stack([gaussian_filter(responses[..., k], blur_sigma)
                         for k in range(responses.shape[-1])], axis=-1)
    h, w, d = smoothed.shape
    features = smoothed.reshape(-1, d)
    if standardize:
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-12)
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit_predict(features)
    return labels.reshape(h, w)


def segment_texton_histograms(image, dictionary, n_clusters, window_size=25, seed=0):
    """K-means on per-pixel texton histograms. Returns ``(labels, texton_map)``."""
    texton_map = dictionary.assign_textons(image)
    hists = compute_texton_histogram(texton_map, dictionary.n_textons, window_size)
    h, w, d = hists.shape
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit_predict(
        hists.reshape(-1, d))
    return labels.reshape(h, w), texton_map


def clustering_accuracy(labels, ground_truth):
    """Pixel accuracy after the best one-to-one matching of cluster ids to
    ground-truth ids (Hungarian algorithm)."""
    k = max(labels.max(), ground_truth.max()) + 1
    confusion = np.zeros((k, k), dtype=np.int64)
    np.add.at(confusion, (labels.ravel(), ground_truth.ravel()), 1)
    rows, cols = linear_sum_assignment(-confusion)
    return confusion[rows, cols].sum() / labels.size
