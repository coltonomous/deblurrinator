"""Recovery-mode barcode deblurring.

A practical module for integrating entropic blind deblurring as a fallback
when standard barcode scanning fails. Optimized for speed over exhaustive
search: relaxed convergence tolerances, fewer inner iterations, and
early stopping on successful decode.

Usage:

    from deblur_recovery import recover_barcode, recover_qr

    # 1D barcode: pass a grayscale numpy array (0-255 or 0.0-1.0)
    result = recover_barcode(blurred_image, m=3)
    if result.success:
        print(result.data)       # decoded string
        print(result.modules)    # recovered binary modules

    # QR code: same interface
    result = recover_qr(blurred_image, m=5, version=1)
    if result.success:
        print(result.data)
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .entropic_deblur import (
    UPCA_N,
    UPCA_QUIET_ZONE,
    QR_QUIET_ZONE,
    upca_symbolic_prior,
    qr_symbolic_prior,
    qr_size,
    upscale,
    downscale_sum,
    _flip,
    _log_partition,
    _sigmoid_r,
    _log_sum_exp_w,
    _softmax_w,
    make_check_fn,
    make_qr_check_fn,
    make_extract_fn_1d,
    make_extract_fn_2d,
    _xt_lam,
)
from scipy.optimize import minimize
from scipy.signal import fftconvolve


# --- Recovery-tuned defaults ---

# Relaxed gradient tolerance: binary signals don't need 1e-10 precision
_GTOL = 1e-6

# Fewer L-BFGS iterations per solve (the early ones do most of the work)
_MAXITER = 200

# Fewer alternations per kernel width (1-2 is usually enough for small widths)
_INNER_ITERS = 2


# --- Result container ---

@dataclass
class DeblurResult:
    """Result of a recovery deblurring attempt."""
    success: bool
    data: Optional[str] = None
    modules: Optional[np.ndarray] = None
    kernel: Optional[np.ndarray] = None
    iterations: int = 0
    kernel_width: int = 0


# --- Fast estimation routines (mirrors entropic_deblur but with relaxed params) ---

def _estimate_image_fast(b, c, r, m, alpha, gtol, maxiter):
    shape_b = b.shape
    c_flip = _flip(c)
    inv_a = 1.0 / alpha

    def objective_and_grad(lam_flat):
        lam = lam_flat.reshape(shape_b)
        v = downscale_sum(fftconvolve(lam, c_flip, mode='same'), m)
        lp = _log_partition(v, r)
        sig = _sigmoid_r(v, r)
        f = -np.sum(b * lam) + 0.5 * inv_a * np.sum(lam**2) + np.sum(lp)
        g = -b + inv_a * lam + fftconvolve(upscale(sig, m), c, mode='same')
        return f, g.ravel()

    res = minimize(objective_and_grad, np.zeros(b.size), jac=True,
                   method='L-BFGS-B', options={'maxiter': maxiter, 'gtol': gtol})

    lam_opt = res.x.reshape(shape_b)
    v_opt = downscale_sum(fftconvolve(lam_opt, c_flip, mode='same'), m)
    return _sigmoid_r(v_opt, r)


def _estimate_kernel_fast(b, x_hat, kernel_shape, m, beta, gtol, maxiter):
    shape_b = b.shape
    signal = upscale(x_hat, m)
    signal_flip = _flip(signal)
    inv_b = 1.0 / beta

    if isinstance(kernel_shape, int):
        kernel_size = kernel_shape
        k_shape = (kernel_shape,)
    else:
        kernel_size = kernel_shape[0] * kernel_shape[1]
        k_shape = kernel_shape

    nu = np.ones(kernel_size) / kernel_size

    def xt(lam):
        return _xt_lam(lam, signal_flip, shape_b, kernel_shape)

    def objective_and_grad(lam_flat):
        lam = lam_flat.reshape(shape_b)
        xtl = xt(lam).ravel()
        lse = _log_sum_exp_w(xtl, nu)
        sm = _softmax_w(xtl, nu).reshape(k_shape)
        f = -np.sum(b * lam) + 0.5 * inv_b * np.sum(lam**2) + lse
        g = -b + inv_b * lam + fftconvolve(signal, sm, mode='same')
        return f, g.ravel()

    res = minimize(objective_and_grad, np.zeros(b.size), jac=True,
                   method='L-BFGS-B', options={'maxiter': maxiter, 'gtol': gtol})

    lam_opt = res.x.reshape(shape_b)
    return _softmax_w(xt(lam_opt).ravel(), nu).reshape(k_shape)


def _fast_blind_deblur(b, r, m, alpha, beta, max_kernel_width,
                       inner_iters, gtol, maxiter, check_fn, extract_fn):
    """Core loop: alternating estimation with early stopping."""
    is_2d = b.ndim == 2
    x_hat = r.copy()
    c_hat = None
    total_iters = 0

    for i in range(1, max_kernel_width // 2 + 1):
        kw = 2 * i + 1
        if kw > min(b.shape):
            break

        kernel_shape = (kw, kw) if is_2d else kw

        for j in range(inner_iters):
            c_hat = _estimate_kernel_fast(b, x_hat, kernel_shape, m, beta, gtol, maxiter)
            x_hat = _estimate_image_fast(b, c_hat, r, m, alpha, gtol, maxiter)
            total_iters += 1

            x_thresh = (x_hat > 0.5).astype(np.float64)
            x_modules = extract_fn(x_thresh)

            if check_fn is not None:
                result = check_fn(x_modules)
                if result:
                    return DeblurResult(
                        success=True, data=result,
                        modules=x_modules, kernel=c_hat,
                        iterations=total_iters, kernel_width=kw,
                    )

    x_modules = extract_fn((x_hat > 0.5).astype(np.float64))
    return DeblurResult(
        success=False, modules=x_modules, kernel=c_hat,
        iterations=total_iters, kernel_width=kw if c_hat is not None else 0,
    )


# --- Public API ---

def recover_barcode(
    blurred_signal: np.ndarray,
    m: int = 3,
    alpha: float = 1e6,
    beta: float = 1e6,
    max_kernel_width: int = 15,
    inner_iters: int = _INNER_ITERS,
    gtol: float = _GTOL,
    maxiter: int = _MAXITER,
) -> DeblurResult:
    """Attempt to recover a 1D UPC-A barcode from a blurred signal.

    Parameters
    ----------
    blurred_signal : 1D array
        The blurred barcode scanline. Values should be in [0, 1] with the
        convention that 0 = black (bar), 1 = white (space). The signal length
        should be approximately (UPCA_N + 2*UPCA_QUIET_ZONE) * m.
    m : int
        Pixels per barcode module in the input signal.
    alpha, beta : float
        Fidelity weights for image and kernel estimation.
    max_kernel_width : int
        Largest odd kernel width to search.
    inner_iters : int
        Alternating iterations per kernel width (default: 2).
    gtol : float
        Gradient tolerance for L-BFGS (default: 1e-6).
    maxiter : int
        Max L-BFGS iterations per solve (default: 200).

    Returns
    -------
    DeblurResult
        .success is True if decoding succeeded, .data has the decoded string.
    """
    b = np.asarray(blurred_signal, dtype=np.float64)
    if b.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {b.shape}")

    # Build the inverted prior (algorithm works in inverted space)
    r_inv = 1.0 - upca_symbolic_prior()
    r_padded = np.concatenate([
        np.zeros(UPCA_QUIET_ZONE), r_inv, np.zeros(UPCA_QUIET_ZONE)
    ])

    check = make_check_fn(m)
    extract = make_extract_fn_1d(UPCA_QUIET_ZONE, UPCA_N)

    return _fast_blind_deblur(
        b, r_padded, m, alpha, beta, max_kernel_width,
        inner_iters, gtol, maxiter, check, extract,
    )


def recover_qr(
    blurred_image: np.ndarray,
    m: int = 5,
    version: Optional[int] = None,
    alpha: float = 1e6,
    beta: float = 1e6,
    max_kernel_width: int = 11,
    inner_iters: int = _INNER_ITERS,
    gtol: float = _GTOL,
    maxiter: int = _MAXITER,
) -> DeblurResult:
    """Attempt to recover a QR code from a blurred image.

    Parameters
    ----------
    blurred_image : 2D array
        The blurred QR code image, values in [0, 1]. The size should be
        approximately (qr_size(version) + 2*QR_QUIET_ZONE) * m in each dim.
    m : int
        Pixels per QR module in the input image.
    version : int or None
        QR version (1-40). Required for building the symbolic prior.
        If None, estimated from image size and m.
    alpha, beta : float
        Fidelity weights for image and kernel estimation.
    max_kernel_width : int
        Largest odd kernel width to search.
    inner_iters : int
        Alternating iterations per kernel width (default: 2).
    gtol : float
        Gradient tolerance for L-BFGS (default: 1e-6).
    maxiter : int
        Max L-BFGS iterations per solve (default: 200).

    Returns
    -------
    DeblurResult
        .success is True if decoding succeeded, .data has the decoded string.
    """
    b = np.asarray(blurred_image, dtype=np.float64)
    if b.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {b.shape}")

    if version is None:
        # Estimate version from image dimensions
        estimated_modules = b.shape[0] / m - 2 * QR_QUIET_ZONE
        version = max(1, round((estimated_modules - 17) / 4))

    size = qr_size(version)
    r_inv = 1.0 - qr_symbolic_prior(version)
    r_padded = np.pad(r_inv, QR_QUIET_ZONE, mode='constant', constant_values=0)

    check = make_qr_check_fn(m)
    extract = make_extract_fn_2d(QR_QUIET_ZONE, size)

    return _fast_blind_deblur(
        b, r_padded, m, alpha, beta, max_kernel_width,
        inner_iters, gtol, maxiter, check, extract,
    )
