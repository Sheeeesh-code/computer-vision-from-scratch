"""Template matching: preprocessing, normalized cross-correlation (NCC) in
three flavours, mean-subtracted NCC and greedy non-maximum suppression.

Image arrays are (H, W) for grayscale or (H, W, C) for colour. Responses are
"valid" correlations, i.e. of shape (H - h + 1, W - w + 1).
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_EPS = 1e-8


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

def rgb2gray(img):
    """Convert an RGB image to grayscale (ITU-R 601 weights), returns uint8."""
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("rgb2gray expects an (H, W, 3) image")
    img = img.astype(np.float64)
    gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return gray.astype(np.uint8)


def pad_zeros(img, pad_top, pad_bottom, pad_left, pad_right):
    """Surround an image with a border of zeros (hand-written ``np.pad``)."""
    h, w = img.shape[:2]
    out_shape = (h + pad_top + pad_bottom, w + pad_left + pad_right) + img.shape[2:]
    out = np.zeros(out_shape, dtype=img.dtype)
    out[pad_top:pad_top + h, pad_left:pad_left + w] = img
    return out


_SOBEL_KERNELS = {
    "h": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float),
    "v": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float),
    "d1": np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=float),
    "d2": np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=float),
}


def gray2grad(img):
    """Gradient maps from Sobel (horizontal / vertical) and Sobel-like
    (45 and 135 degrees) kernels, computed as true convolutions with a
    zero-padded border.

    Returns ``(grad_h, grad_v, grad_d1, grad_d2)``, each of shape (H, W).
    """
    img = pad_zeros(img.astype(np.float64), 1, 1, 1, 1)
    windows = sliding_window_view(img, (3, 3))  # (H, W, 3, 3)
    return tuple(
        np.einsum("ijkl,kl->ij", windows, np.flip(_SOBEL_KERNELS[k]))
        for k in ("h", "v", "d1", "d2")
    )


# --------------------------------------------------------------------------
# Normalized cross-correlation
# --------------------------------------------------------------------------

def _output_shape(img, template):
    return img.shape[0] - template.shape[0] + 1, img.shape[1] - template.shape[1] + 1


def ncc_naive(img, template):
    """NCC with 4 (grayscale) or 5 (colour) nested loops. Reference version."""
    img = img.astype(np.float64)
    template = template.astype(np.float64)
    Ho, Wo = _output_shape(img, template)
    Hk, Wk = template.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1

    template_norm = max(np.sqrt(np.sum(template ** 2)), _EPS)
    response = np.zeros((Ho, Wo))

    for i in range(Ho):
        for j in range(Wo):
            dot = 0.0
            window_sq = 0.0
            for k in range(Hk):
                for l in range(Wk):
                    if img.ndim == 3:
                        for c in range(channels):
                            f, p = template[k, l, c], img[i + k, j + l, c]
                            dot += f * p
                            window_sq += p ** 2
                    else:
                        f, p = template[k, l], img[i + k, j + l]
                        dot += f * p
                        window_sq += p ** 2
            response[i, j] = dot / (template_norm * max(np.sqrt(window_sq), _EPS))
    return response


def ncc_fast(img, template):
    """NCC with two loops: the template loops become one element-wise product."""
    img = img.astype(np.float64)
    template = template.astype(np.float64)
    Ho, Wo = _output_shape(img, template)
    Hk, Wk = template.shape[:2]

    template_norm = max(np.sqrt(np.sum(template ** 2)), _EPS)
    response = np.zeros((Ho, Wo))
    for i in range(Ho):
        for j in range(Wo):
            window = img[i:i + Hk, j:j + Wk]
            dot = np.sum(window * template)
            window_norm = np.sqrt(np.sum(window ** 2))
            response[i, j] = dot / (template_norm * max(window_norm, _EPS))
    return response


def _im2col(img, Hk, Wk):
    """Stack every (Hk, Wk, C) window of ``img`` as a row: (Ho*Wo, Hk*Wk*C)."""
    windows = sliding_window_view(img, (Hk, Wk), axis=(0, 1))
    if img.ndim == 3:  # (Ho, Wo, C, Hk, Wk) -> (Ho, Wo, Hk, Wk, C)
        windows = windows.transpose(0, 1, 3, 4, 2)
    Ho, Wo = windows.shape[:2]
    return windows.reshape(Ho * Wo, -1), Ho, Wo


def ncc_matrix(img, template):
    """NCC as one matrix product ``X = P @ F`` (im2col formulation).

    Fastest of the three, but materialises an (Ho*Wo, Hk*Wk*C) matrix, so it
    needs memory proportional to image area times template area.
    """
    img = img.astype(np.float64)
    template = template.astype(np.float64)
    Hk, Wk = template.shape[:2]

    P, Ho, Wo = _im2col(img, Hk, Wk)
    F = template.reshape(-1, 1)

    dot = (P @ F).ravel()
    denom = np.linalg.norm(F) * np.linalg.norm(P, axis=1)
    denom[denom == 0] = _EPS
    return (dot / denom).reshape(Ho, Wo)


def ncc_mean_subtracted(img, template):
    """Mean-subtracted NCC (equivalent to OpenCV ``TM_CCOEFF_NORMED``).

    The mean of the template and of every window is removed, separately for
    each colour channel, before normalising.
    """
    img = img.astype(np.float64)
    template = template.astype(np.float64)
    Ho, Wo = _output_shape(img, template)
    Hk, Wk = template.shape[:2]
    axes = (0, 1)

    template_ms = template - template.mean(axis=axes, keepdims=True)
    template_norm = max(np.sqrt(np.sum(template_ms ** 2)), _EPS)

    response = np.zeros((Ho, Wo))
    for i in range(Ho):
        for j in range(Wo):
            window = img[i:i + Hk, j:j + Wk]
            window = window - window.mean(axis=axes, keepdims=True)
            dot = np.sum(window * template_ms)
            window_norm = np.sqrt(np.sum(window ** 2))
            response[i, j] = dot / (template_norm * max(window_norm, _EPS))
    return response


# --------------------------------------------------------------------------
# Non-maximum suppression
# --------------------------------------------------------------------------

def non_max_suppression(response, suppress_range, threshold=None):
    """Greedy non-maximum suppression on a response map.

    Repeatedly pick the global maximum, record it, then zero a
    ``(2*H_range+1) x (2*W_range+1)`` window around it.

    Parameters
    ----------
    response : 2-D array
    suppress_range : (H_range, W_range)
    threshold : values strictly below it are discarded first

    Returns
    -------
    A sparse array of the same shape holding only the retained maxima.
    """
    res = response.copy()
    if threshold is not None:
        res[res < threshold] = 0
    h_range, w_range = suppress_range
    peaks = np.zeros_like(res)

    while np.max(res) > 0:
        i, j = np.unravel_index(np.argmax(res), res.shape)
        peaks[i, j] = res[i, j]
        res[max(0, i - h_range):i + h_range + 1,
            max(0, j - w_range):j + w_range + 1] = 0
    return peaks
