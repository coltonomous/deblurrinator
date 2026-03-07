"""Blind deblurring of barcodes via Kullback-Leibler divergence.

Implementation of Algorithm 1 from:
  "Blind Deblurring of Barcodes via Kullback-Leibler Divergence"
  Rioux, Scarvelis, Choksi, Hoheisel, Marechal
  IEEE TPAMI, Vol. 43, No. 1, January 2021

The method models a barcode as N independent Bernoulli random variables and
finds the probability distribution over all 2^N possible barcodes that best
explains the observed blurry signal. KL divergence from a symbolic prior
regularizes the solution, and Fenchel-Rockafellar duality reduces the
exponentially large primal problem to a tractable dual solved with L-BFGS.

Supports both 1D barcodes (UPC-A) and 2D barcodes (QR codes).
"""

import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.special import expit


# ============================================================
# UPC-A Encoding
# ============================================================

# L-code patterns (barcode standard: 1=dark bar, 0=light space)
_L_PATTERNS = {
    '0': (0,0,0,1,1,0,1), '1': (0,0,1,1,0,0,1),
    '2': (0,0,1,0,0,1,1), '3': (0,1,1,1,1,0,1),
    '4': (0,1,0,0,0,1,1), '5': (0,1,1,0,0,0,1),
    '6': (0,1,0,1,1,1,1), '7': (0,1,1,1,0,1,1),
    '8': (0,1,1,0,1,1,1), '9': (0,0,0,1,0,1,1),
}
_R_PATTERNS = {k: tuple(1 - b for b in v) for k, v in _L_PATTERNS.items()}

UPCA_N = 95
_START_GUARD = (1, 0, 1)
_MIDDLE_GUARD = (0, 1, 0, 1, 0)
_END_GUARD = (1, 0, 1)


def compute_check_digit(digits_11):
    """Compute UPC-A check digit from first 11 digits."""
    odd = sum(int(digits_11[i]) for i in range(0, 11, 2))
    even = sum(int(digits_11[i]) for i in range(1, 11, 2))
    return str((10 - (odd * 3 + even) % 10) % 10)


def encode_upca(digits):
    """
    Encode a UPC-A barcode as a 95-element binary array.

    Args:
        digits: string of 11 or 12 digits (check digit computed if 11).

    Returns:
        x: array of shape (95,) with values in {0.0, 1.0}.
           Convention: 1 = white (space), 0 = black (bar).
    """
    if len(digits) == 11:
        digits = digits + compute_check_digit(digits)
    assert len(digits) == 12, f"Expected 11 or 12 digits, got {len(digits)}"

    pattern = list(_START_GUARD)
    for d in digits[:6]:
        pattern.extend(_L_PATTERNS[d])
    pattern.extend(_MIDDLE_GUARD)
    for d in digits[6:]:
        pattern.extend(_R_PATTERNS[d])
    pattern.extend(_END_GUARD)

    # Invert: barcode standard 1=dark -> image 0=black
    return np.array([1 - b for b in pattern], dtype=np.float64)


def upca_symbolic_prior():
    """
    Symbolic prior r for UPC-A barcodes.

    r[i] = probability that module i is white.
    Fixed modules: 0 (black) or 1 (white). Data modules: 0.5.
    """
    r = np.full(UPCA_N, 0.5)

    # Start guard: bar-space-bar -> image: 0, 1, 0
    r[0], r[1], r[2] = 0.0, 1.0, 0.0

    # Middle guard: space-bar-space-bar-space -> image: 1, 0, 1, 0, 1
    r[45], r[46], r[47], r[48], r[49] = 1.0, 0.0, 1.0, 0.0, 1.0

    # End guard: bar-space-bar -> image: 0, 1, 0
    r[92], r[93], r[94] = 0.0, 1.0, 0.0

    return r


# ============================================================
# QR Code Encoding
# ============================================================

# Alignment pattern center positions per QR version (versions 2-40)
_QR_ALIGNMENT_POSITIONS = {
    2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30], 6: [6, 34],
    7: [6, 22, 38], 8: [6, 24, 42], 9: [6, 26, 46], 10: [6, 28, 50],
    11: [6, 30, 54], 12: [6, 32, 58], 13: [6, 34, 62], 14: [6, 26, 46, 66],
    15: [6, 26, 48, 70], 16: [6, 26, 50, 74], 17: [6, 30, 54, 78],
    18: [6, 30, 56, 82], 19: [6, 30, 58, 86], 20: [6, 34, 62, 90],
    21: [6, 28, 50, 72, 94], 22: [6, 26, 50, 74, 98],
    23: [6, 30, 54, 78, 102], 24: [6, 28, 54, 80, 106],
    25: [6, 32, 58, 84, 110], 26: [6, 30, 58, 86, 114],
    27: [6, 34, 62, 90, 118], 28: [6, 26, 50, 74, 98, 122],
    29: [6, 30, 54, 78, 102, 126], 30: [6, 26, 52, 78, 104, 130],
    31: [6, 30, 56, 82, 108, 134], 32: [6, 34, 60, 86, 112, 138],
    33: [6, 30, 58, 86, 114, 142], 34: [6, 34, 62, 90, 118, 146],
    35: [6, 30, 54, 78, 102, 126, 150], 36: [6, 24, 50, 76, 102, 128, 154],
    37: [6, 28, 54, 80, 106, 132, 158], 38: [6, 32, 58, 84, 110, 136, 162],
    39: [6, 26, 54, 82, 110, 138, 166], 40: [6, 30, 58, 86, 114, 142, 170],
}

# 7x7 finder pattern (1=dark, 0=light in QR standard)
_FINDER_PATTERN = np.array([
    [1,1,1,1,1,1,1],
    [1,0,0,0,0,0,1],
    [1,0,1,1,1,0,1],
    [1,0,1,1,1,0,1],
    [1,0,1,1,1,0,1],
    [1,0,0,0,0,0,1],
    [1,1,1,1,1,1,1],
], dtype=np.float64)

# 5x5 alignment pattern (1=dark, 0=light in QR standard)
_ALIGNMENT_PATTERN = np.array([
    [1,1,1,1,1],
    [1,0,0,0,1],
    [1,0,1,0,1],
    [1,0,0,0,1],
    [1,1,1,1,1],
], dtype=np.float64)


def qr_size(version):
    """QR code side length in modules for a given version (1-40)."""
    return 17 + 4 * version


def qr_symbolic_prior(version):
    """
    Symbolic prior for QR codes of a given version.

    Returns r as a 2D array of shape (size, size).
    r[i,j] = probability that module (i,j) is white (image convention).
    Fixed modules: 0.0 (black) or 1.0 (white). Data modules: 0.5.

    Fixed modules include: finder patterns, separators, timing patterns,
    dark module, alignment patterns. Format/version info are treated as
    data (0.5) since they vary with content.
    """
    size = qr_size(version)
    r = np.full((size, size), 0.5)

    # -- Finder patterns (top-left, top-right, bottom-left) --
    # In image convention: dark=0, light=1, so invert the pattern
    fp_img = 1.0 - _FINDER_PATTERN

    # Top-left finder
    r[0:7, 0:7] = fp_img
    # Top-right finder
    r[0:7, size-7:size] = fp_img
    # Bottom-left finder
    r[size-7:size, 0:7] = fp_img

    # -- Separators (1 module white border around finders) --
    # Top-left separator
    r[7, 0:8] = 1.0  # bottom edge
    r[0:7, 7] = 1.0  # right edge
    # Top-right separator
    r[7, size-8:size] = 1.0  # bottom edge
    r[0:7, size-8] = 1.0  # left edge
    # Bottom-left separator
    r[size-8, 0:8] = 1.0  # top edge
    r[size-7:size, 7] = 1.0  # right edge

    # -- Timing patterns (alternating dark/light on row 6 and col 6) --
    for i in range(8, size - 8):
        val = 1.0 if i % 2 == 1 else 0.0  # even=dark(0), odd=light(1)
        r[6, i] = val  # horizontal timing
        r[i, 6] = val  # vertical timing

    # -- Dark module (always dark, at position (4*version + 9, 8)) --
    r[4 * version + 9, 8] = 0.0

    # -- Alignment patterns (version 2+) --
    if version >= 2 and version in _QR_ALIGNMENT_POSITIONS:
        positions = _QR_ALIGNMENT_POSITIONS[version]
        ap_img = 1.0 - _ALIGNMENT_PATTERN
        for row_c in positions:
            for col_c in positions:
                # Skip if overlapping with finder patterns or separators
                r0, r1 = row_c - 2, row_c + 3
                c0, c1 = col_c - 2, col_c + 3
                # Check overlap with top-left finder+separator (0..7, 0..7)
                if r0 < 8 and c0 < 8:
                    continue
                # Check overlap with top-right finder+separator (0..7, size-8..size-1)
                if r0 < 8 and c1 > size - 8:
                    continue
                # Check overlap with bottom-left finder+separator (size-8..size-1, 0..7)
                if r1 > size - 8 and c0 < 8:
                    continue
                r[r0:r1, c0:c1] = ap_img

    return r


def encode_qr(data, version=None, error_correction='M'):
    """
    Encode data into a QR code module array.

    Args:
        data:             string data to encode
        version:          QR version (1-40), auto-selected if None
        error_correction: 'L', 'M', 'Q', or 'H'

    Returns:
        modules: 2D array of shape (size, size), image convention (1=white, 0=black)
        version: the QR version used
    """
    import qrcode

    ec_map = {
        'L': qrcode.constants.ERROR_CORRECT_L,
        'M': qrcode.constants.ERROR_CORRECT_M,
        'Q': qrcode.constants.ERROR_CORRECT_Q,
        'H': qrcode.constants.ERROR_CORRECT_H,
    }

    qr = qrcode.QRCode(
        version=version,
        error_correction=ec_map[error_correction],
        box_size=1,
        border=0,
    )
    qr.add_data(data)
    qr.make(fit=(version is None))

    # qr.modules is a list of lists of bools (True=dark)
    modules = np.array(qr.modules, dtype=np.float64)
    # Invert to image convention: dark(True/1) -> 0, light(False/0) -> 1
    modules = 1.0 - modules

    return modules, qr.version


# ============================================================
# Signal Operations (dimension-agnostic)
# ============================================================

def upscale(x, m):
    """Repeat each element m times along each axis."""
    if x.ndim == 1:
        return np.repeat(x, m)
    elif x.ndim == 2:
        return np.kron(x, np.ones((m, m)))
    else:
        raise ValueError(f"upscale supports 1D or 2D, got {x.ndim}D")


def downscale_sum(y, m):
    """Sum blocks of m elements along each axis (adjoint of upscale)."""
    if y.ndim == 1:
        n = len(y) // m
        return y[:n * m].reshape(n, m).sum(axis=1)
    elif y.ndim == 2:
        nh = y.shape[0] // m
        nw = y.shape[1] // m
        return y[:nh * m, :nw * m].reshape(nh, m, nw, m).sum(axis=(1, 3))
    else:
        raise ValueError(f"downscale_sum supports 1D or 2D, got {y.ndim}D")


def _flip(x):
    """Flip array along all axes."""
    return x[tuple(slice(None, None, -1) for _ in range(x.ndim))]


# ============================================================
# Numerically Stable Helpers
# ============================================================

def _log_partition(z, r):
    """
    log(1 - r + r*exp(z)), element-wise, numerically stable.
    KL conjugate contribution per Bernoulli variable (eq 13).
    """
    out = np.empty_like(z)

    m0, m1 = r == 0, r == 1
    mf = ~m0 & ~m1

    out[m0] = 0.0
    out[m1] = z[m1]

    if np.any(mf):
        zf, rf = z[mf], r[mf]
        zmax = np.maximum(0.0, zf)
        out[mf] = zmax + np.log((1 - rf) * np.exp(-zmax) + rf * np.exp(zf - zmax))

    return out


def _sigmoid_r(z, r):
    """
    r*exp(z) / (1 - r + r*exp(z)), element-wise, numerically stable.
    Gradient of _log_partition; recovers the image estimate (eq 15).
    """
    out = np.empty_like(z)

    m0, m1 = r == 0, r == 1
    mf = ~m0 & ~m1

    out[m0] = 0.0
    out[m1] = 1.0

    if np.any(mf):
        zf, rf = z[mf], r[mf]
        out[mf] = expit(zf + np.log(rf / (1 - rf)))

    return out


def _log_sum_exp_w(z, w):
    """Weighted log-sum-exp: log(sum(w * exp(z))) for w >= 0."""
    mask = w > 0
    zm, wm = z[mask], w[mask]
    c = np.max(zm)
    return c + np.log(np.sum(wm * np.exp(zm - c)))


def _softmax_w(z, w):
    """Weighted softmax: w * exp(z) / sum(w * exp(z))."""
    out = np.zeros_like(z)
    mask = w > 0
    zm, wm = z[mask], w[mask]
    c = np.max(zm)
    e = wm * np.exp(zm - c)
    out[mask] = e / np.sum(e)
    return out


# ============================================================
# Image Estimation (Section 2.1) — dimension-agnostic
# ============================================================

def estimate_image(b, c, r, m, alpha):
    """
    Solve the image estimation subproblem via Fenchel-Rockafellar dual.

    Works for both 1D signals and 2D images.

    Args:
        b:     blurred signal (shape matches upscaled domain)
        c:     PSF estimate (odd-length/odd-sized kernel)
        r:     symbolic prior (shape N or (Nh, Nw))
        m:     upscaling factor
        alpha: fidelity weight

    Returns:
        x_hat: barcode/QR estimate, values in [0, 1]
    """
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
                   method='L-BFGS-B', options={'maxiter': 500, 'gtol': 1e-10})

    lam_opt = res.x.reshape(shape_b)
    v_opt = downscale_sum(fftconvolve(lam_opt, c_flip, mode='same'), m)
    return _sigmoid_r(v_opt, r)


# ============================================================
# Kernel Estimation (Section 2.2) — dimension-agnostic
# ============================================================

def _xt_lam(lam, signal_flip, Nm_shape, kernel_shape):
    """
    Adjoint of X (convolution with signal) applied to lam.

    Works for 1D (kernel_shape is int or (kw,)) and 2D (kernel_shape is (kh,kw)).
    """
    full = fftconvolve(lam, signal_flip, mode='full')

    if lam.ndim == 1:
        kw = kernel_shape if isinstance(kernel_shape, int) else kernel_shape[0]
        Nm = Nm_shape if isinstance(Nm_shape, int) else Nm_shape[0]
        offset = (kw - 1) // 2
        start = Nm - 1 - offset
        return full[start : start + kw].copy()
    else:
        if isinstance(kernel_shape, int):
            kh, kw = kernel_shape, kernel_shape
        else:
            kh, kw = kernel_shape
        if isinstance(Nm_shape, int):
            Nmh, Nmw = Nm_shape, Nm_shape
        else:
            Nmh, Nmw = Nm_shape

        off_h = (kh - 1) // 2
        off_w = (kw - 1) // 2
        start_h = Nmh - 1 - off_h
        start_w = Nmw - 1 - off_w
        return full[start_h : start_h + kh, start_w : start_w + kw].copy()


def estimate_kernel(b, x_hat, kernel_shape, m, beta):
    """
    Solve the kernel estimation subproblem via Fenchel-Rockafellar dual.

    Works for both 1D and 2D.

    Args:
        b:            blurred signal
        x_hat:        barcode/QR estimate
        kernel_shape: int for 1D width, or (kh, kw) tuple for 2D
        m:            upscaling factor
        beta:         fidelity weight

    Returns:
        c_hat: estimated PSF (sums to 1)
    """
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

    Nm_shape = shape_b
    nu = np.ones(kernel_size) / kernel_size

    def xt(lam):
        return _xt_lam(lam, signal_flip, Nm_shape, kernel_shape)

    def objective_and_grad(lam_flat):
        lam = lam_flat.reshape(shape_b)
        xtl = xt(lam).ravel()

        lse = _log_sum_exp_w(xtl, nu)
        sm = _softmax_w(xtl, nu).reshape(k_shape)

        f = -np.sum(b * lam) + 0.5 * inv_b * np.sum(lam**2) + lse
        g = -b + inv_b * lam + fftconvolve(signal, sm, mode='same')

        return f, g.ravel()

    res = minimize(objective_and_grad, np.zeros(b.size), jac=True,
                   method='L-BFGS-B', options={'maxiter': 500, 'gtol': 1e-10})

    lam_opt = res.x.reshape(shape_b)
    xtl = xt(lam_opt).ravel()
    return _softmax_w(xtl, nu).reshape(k_shape)


# ============================================================
# Algorithm 1: Entropic Blind Deblurring (dimension-agnostic)
# ============================================================

UPCA_QUIET_ZONE = 9
QR_QUIET_ZONE = 4


def prepare_signal_1d(x, m, kernel, noise_var=0.0):
    """
    Prepare a blurred 1D test signal with quiet zone padding and inversion.

    Args:
        x:         barcode in image convention (1=white, 0=black), length N
        m:         upscaling factor
        kernel:    1D blur kernel
        noise_var: additive Gaussian noise variance

    Returns:
        b_inv: inverted blurred signal with quiet zone
        r_inv: inverted prior with quiet zone
    """
    qz = UPCA_QUIET_ZONE
    r = upca_symbolic_prior()

    x_inv = 1.0 - x
    r_inv = 1.0 - r

    x_padded = np.concatenate([np.zeros(qz), x_inv, np.zeros(qz)])
    r_padded = np.concatenate([np.zeros(qz), r_inv, np.zeros(qz)])

    signal = upscale(x_padded, m)
    b = blur_signal(signal, kernel, noise_var)

    return b, r_padded


def prepare_signal_2d(x, m, kernel, noise_var=0.0, version=None):
    """
    Prepare a blurred 2D QR signal with quiet zone padding and inversion.

    Args:
        x:         QR code in image convention (1=white, 0=black), shape (N, N)
        m:         upscaling factor
        kernel:    2D blur kernel
        noise_var: additive Gaussian noise variance
        version:   QR version (needed for prior; inferred from size if None)

    Returns:
        b_inv: inverted blurred signal with quiet zone
        r_inv: inverted prior with quiet zone
    """
    qz = QR_QUIET_ZONE
    size = x.shape[0]
    if version is None:
        version = (size - 17) // 4

    r = qr_symbolic_prior(version)

    x_inv = 1.0 - x
    r_inv = 1.0 - r

    x_padded = np.pad(x_inv, qz, mode='constant', constant_values=0)
    r_padded = np.pad(r_inv, qz, mode='constant', constant_values=0)

    signal = upscale(x_padded, m)
    b = blur_signal(signal, kernel, noise_var)

    return b, r_padded


# Backward-compatible alias
prepare_signal = prepare_signal_1d


def entropic_blind_deblur(b, r, m, alpha=1e6, beta=1e6,
                          max_kernel_width=None, inner_iters=5,
                          check_fn=None, extract_fn=None,
                          verbose=True):
    """
    Blind deblurring via alternating entropy-regularized optimization.

    Works for both 1D signals and 2D images. Kernel shape is always square
    for 2D (kw x kw) or scalar for 1D.

    Args:
        b:                blurred signal (1D or 2D, inverted, with quiet zone)
        r:                symbolic prior (inverted, with quiet zone)
        m:                upscaling factor
        alpha:            image estimation fidelity weight
        beta:             kernel estimation fidelity weight
        max_kernel_width: largest kernel width to try (default: smallest dim // 2)
        inner_iters:      alternating iterations per kernel width
        check_fn:         callable(x_modules) -> str or None
                          (operates on the original-size modules in image convention)
        extract_fn:       callable(x_hat_thresh) -> x_modules
                          strips quiet zone and inverts back to image convention.
                          If None, uses default 1D UPC-A extraction.
        verbose:          print progress

    Returns:
        x_hat:   extracted module estimate (image convention)
        c_hat:   final kernel estimate
        success: whether a readable barcode was recovered
    """
    is_2d = b.ndim == 2

    if max_kernel_width is None:
        max_kernel_width = min(b.shape) // 2

    if extract_fn is None:
        qz = UPCA_QUIET_ZONE
        n_modules = UPCA_N
        def extract_fn(x_thresh):
            return 1.0 - x_thresh[qz : qz + n_modules]

    x_hat = r.copy()
    c_hat = None

    for i in range(1, max_kernel_width // 2 + 1):
        kw = 2 * i + 1
        if kw > min(b.shape):
            break

        kernel_shape = (kw, kw) if is_2d else kw

        for j in range(inner_iters):
            c_hat = estimate_kernel(b, x_hat, kernel_shape, m, beta)
            x_hat = estimate_image(b, c_hat, r, m, alpha)

            x_thresh = (x_hat > 0.5).astype(np.float64)

            if check_fn is not None:
                x_modules = extract_fn(x_thresh)
                result = check_fn(x_modules)
                if result:
                    if verbose:
                        print(f"Decoded at kernel_width={kw}, iter={j+1}: {result}")
                    return x_modules, c_hat, True

        if verbose:
            print(f"kernel_width={kw}: not decoded")

    if verbose:
        print("Terminated without decoding")
    x_modules = extract_fn((x_hat > 0.5).astype(np.float64))
    return x_modules, c_hat, False


# ============================================================
# Readability Checks
# ============================================================

def make_check_fn(m):
    """Create a 1D barcode readability checker using pyzbar."""
    try:
        from pyzbar.pyzbar import decode
    except ImportError:
        print("pyzbar not available; readability checking disabled")
        return None

    def check(x):
        qz = np.ones(9 * m)
        img_1d = np.concatenate([qz, upscale(x, m), qz])
        img_1d = (img_1d * 255).astype(np.uint8)
        img_2d = np.tile(img_1d.reshape(1, -1), (50, 1))
        detected = decode(img_2d)
        if detected:
            return detected[0].data.decode()
        return None

    return check


def make_qr_check_fn(m):
    """Create a QR code readability checker using pyzbar."""
    try:
        from pyzbar.pyzbar import decode
    except ImportError:
        print("pyzbar not available; readability checking disabled")
        return None

    def check(x):
        # Add quiet zone (4 modules of white)
        padded = np.pad(x, QR_QUIET_ZONE, mode='constant', constant_values=1.0)
        img = upscale(padded, m)
        img = (img * 255).astype(np.uint8)
        detected = decode(img)
        if detected:
            return detected[0].data.decode()
        return None

    return check


# ============================================================
# Extract Functions
# ============================================================

def make_extract_fn_1d(quiet_zone, n_modules):
    """Create extraction function for 1D barcodes."""
    def extract(x_thresh):
        return 1.0 - x_thresh[quiet_zone : quiet_zone + n_modules]
    return extract


def make_extract_fn_2d(quiet_zone, size):
    """Create extraction function for 2D codes (QR)."""
    def extract(x_thresh):
        return 1.0 - x_thresh[quiet_zone : quiet_zone + size,
                               quiet_zone : quiet_zone + size]
    return extract


# ============================================================
# Synthetic Blurring
# ============================================================

def gaussian_kernel_1d(width, sigma):
    """Create a normalized 1D Gaussian kernel."""
    t = np.arange(width) - width // 2
    k = np.exp(-t**2 / (2 * sigma**2))
    return k / k.sum()


def gaussian_kernel_2d(width, sigma):
    """Create a normalized 2D isotropic Gaussian kernel."""
    t = np.arange(width) - width // 2
    g = np.exp(-t**2 / (2 * sigma**2))
    k = np.outer(g, g)
    return k / k.sum()


def box_kernel_1d(width):
    """Create a normalized 1D box kernel."""
    return np.ones(width) / width


def box_kernel_2d(width):
    """Create a normalized 2D box kernel."""
    return np.ones((width, width)) / (width * width)


def motion_kernel(width, angle_deg=0):
    """
    Create a normalized 1-pixel-wide motion blur kernel.

    Args:
        width: kernel size (odd integer)
        angle_deg: motion direction in degrees (0=horizontal, 90=vertical)

    Returns:
        k: (width, width) normalized kernel
    """
    k = np.zeros((width, width))
    center = width // 2
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    for i in range(width):
        t = i - center
        x = center + t * cos_a
        y = center + t * sin_a
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < width and 0 <= yi < width:
            k[yi, xi] = 1.0

    return k / k.sum()


def blur_signal(signal, kernel, noise_var=0.0):
    """Apply blur kernel and optional Gaussian noise."""
    b = fftconvolve(signal, kernel, mode='same')
    if noise_var > 0:
        b += np.random.normal(0, np.sqrt(noise_var), b.shape)
        b = np.clip(b, 0, 1)
    return b


# ============================================================
# Demo: 1D UPC-A
# ============================================================

def demo(blur_width=21, sigma=1.0, m=5, noise_var=0.0,
         alpha=1e6, beta=1e6):
    """Run blind deblurring on a synthetic UPC-A barcode."""
    import random

    digits = ''.join([str(random.randint(0, 9)) for _ in range(11)])
    digits += compute_check_digit(digits)
    print(f"Original barcode: {digits}")

    x = encode_upca(digits)
    kernel = gaussian_kernel_1d(blur_width, sigma)
    b, r_inv = prepare_signal_1d(x, m, kernel, noise_var)
    print(f"Applied Gaussian blur: width={blur_width}, sigma={sigma:.1f}")
    if noise_var > 0:
        print(f"Added Gaussian noise: variance={noise_var}")

    check = make_check_fn(m)
    extract = make_extract_fn_1d(UPCA_QUIET_ZONE, UPCA_N)

    print("Running entropic blind deblurring...")
    x_hat, c_hat, success = entropic_blind_deblur(
        b, r_inv, m, alpha=alpha, beta=beta,
        check_fn=check, extract_fn=extract, verbose=True
    )

    if success:
        print("SUCCESS: Barcode recovered and readable!")
    else:
        print("FAILED: Could not recover readable barcode")
        if check:
            result = check(x_hat)
            if result:
                print(f"  (but final decode returned: {result})")

    accuracy = np.mean(x_hat == x) * 100
    print(f"Module accuracy: {accuracy:.1f}% ({int(accuracy * UPCA_N / 100)}/{UPCA_N})")

    return x, x_hat, b, c_hat, kernel


# ============================================================
# Demo: 2D QR Code
# ============================================================

def demo_qr(data="HELLO WORLD", blur_width=5, sigma=1.0, m=5,
            noise_var=0.0, alpha=1e6, beta=1e6,
            error_correction='M', version=None):
    """Run blind deblurring on a synthetic QR code."""
    print(f"QR data: {data!r}")

    x, ver = encode_qr(data, version=version, error_correction=error_correction)
    size = qr_size(ver)
    print(f"QR version {ver}, size {size}x{size}")

    kernel = gaussian_kernel_2d(blur_width, sigma)
    b, r_inv = prepare_signal_2d(x, m, kernel, noise_var, version=ver)
    print(f"Applied 2D Gaussian blur: width={blur_width}, sigma={sigma:.1f}")
    print(f"Blurred signal shape: {b.shape}")
    if noise_var > 0:
        print(f"Added Gaussian noise: variance={noise_var}")

    check = make_qr_check_fn(m)
    extract = make_extract_fn_2d(QR_QUIET_ZONE, size)

    print("Running entropic blind deblurring...")
    x_hat, c_hat, success = entropic_blind_deblur(
        b, r_inv, m, alpha=alpha, beta=beta,
        check_fn=check, extract_fn=extract, verbose=True
    )

    if success:
        print("SUCCESS: QR code recovered and readable!")
    else:
        print("FAILED: Could not recover readable QR code")
        if check:
            result = check(x_hat)
            if result:
                print(f"  (but final decode returned: {result})")

    accuracy = np.mean(x_hat == x) * 100
    total = size * size
    print(f"Module accuracy: {accuracy:.1f}% ({int(accuracy * total / 100)}/{total})")

    return x, x_hat, b, c_hat, kernel


if __name__ == '__main__':
    demo()
