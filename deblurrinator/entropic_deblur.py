"""Blind deblurring of barcodes via Kullback-Leibler divergence.

Implements Algorithm 1 from Rioux et al., "Blind Deblurring of Barcodes via
Kullback-Leibler Divergence" (IEEE TPAMI, 2021). Supports 1D (UPC-A) and 2D (QR).

Core idea: model a barcode as N independent Bernoulli RVs, then find the
distribution over all 2^N possible barcodes that best explains the blurry
observation. KL divergence from a symbolic prior regularizes the solution.
Fenchel-Rockafellar duality collapses the exponential primal into an Nm-variable
dual solved by L-BFGS.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.special import expit


# --- UPC-A encoding ---

_L_PATTERNS = {
    '0': (0,0,0,1,1,0,1), '1': (0,0,1,1,0,0,1),
    '2': (0,0,1,0,0,1,1), '3': (0,1,1,1,1,0,1),
    '4': (0,1,0,0,0,1,1), '5': (0,1,1,0,0,0,1),
    '6': (0,1,0,1,1,1,1), '7': (0,1,1,1,0,1,1),
    '8': (0,1,1,0,1,1,1), '9': (0,0,0,1,0,1,1),
}
_R_PATTERNS = {k: tuple(1 - b for b in v) for k, v in _L_PATTERNS.items()}

UPCA_N = 95  # Total modules in a UPC-A barcode (3+42+5+42+3)
_START_GUARD = (1, 0, 1)
_MIDDLE_GUARD = (0, 1, 0, 1, 0)
_END_GUARD = (1, 0, 1)


def compute_check_digit(digits_11):
    odd = sum(int(digits_11[i]) for i in range(0, 11, 2))
    even = sum(int(digits_11[i]) for i in range(1, 11, 2))
    return str((10 - (odd * 3 + even) % 10) % 10)


def encode_upca(digits):
    """Encode UPC-A digits into a 95-element binary array (1=white, 0=black)."""
    if not isinstance(digits, str) or not digits.isdigit():
        raise ValueError(f"digits must be a string of numeric characters, got {digits!r}")
    if len(digits) == 11:
        digits = digits + compute_check_digit(digits)
    if len(digits) != 12:
        raise ValueError(f"Expected 11 or 12 digits, got {len(digits)}")

    pattern = list(_START_GUARD)
    for d in digits[:6]:
        pattern.extend(_L_PATTERNS[d])
    pattern.extend(_MIDDLE_GUARD)
    for d in digits[6:]:
        pattern.extend(_R_PATTERNS[d])
    pattern.extend(_END_GUARD)

    return np.array([1 - b for b in pattern], dtype=np.float64)


def upca_symbolic_prior():
    """Prior r[i] = P(module i is white). Fixed guard modules pinned, data = 0.5."""
    r = np.full(UPCA_N, 0.5)
    r[0], r[1], r[2] = 0.0, 1.0, 0.0                          # start guard
    r[45], r[46], r[47], r[48], r[49] = 1.0, 0.0, 1.0, 0.0, 1.0  # middle guard
    r[92], r[93], r[94] = 0.0, 1.0, 0.0                        # end guard
    return r


# --- QR code encoding ---

# Alignment pattern centers per version (from the spec)
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

_FINDER_PATTERN = np.array([
    [1,1,1,1,1,1,1],
    [1,0,0,0,0,0,1],
    [1,0,1,1,1,0,1],
    [1,0,1,1,1,0,1],
    [1,0,1,1,1,0,1],
    [1,0,0,0,0,0,1],
    [1,1,1,1,1,1,1],
], dtype=np.float64)

_ALIGNMENT_PATTERN = np.array([
    [1,1,1,1,1],
    [1,0,0,0,1],
    [1,0,1,0,1],
    [1,0,0,0,1],
    [1,1,1,1,1],
], dtype=np.float64)


def qr_size(version):
    return 17 + 4 * version


def qr_symbolic_prior(version):
    """Build the symbolic prior for a QR code. Fixed structural modules are
    pinned to 0 or 1; everything else (data, format info) is 0.5."""
    size = qr_size(version)
    r = np.full((size, size), 0.5)

    # Finder patterns (inverted: dark=0 in image convention)
    fp = 1.0 - _FINDER_PATTERN
    r[0:7, 0:7] = fp
    r[0:7, size-7:size] = fp
    r[size-7:size, 0:7] = fp

    # Separators (white border around finders)
    r[7, 0:8] = 1.0
    r[0:7, 7] = 1.0
    r[7, size-8:size] = 1.0
    r[0:7, size-8] = 1.0
    r[size-8, 0:8] = 1.0
    r[size-7:size, 7] = 1.0

    # Timing patterns (alternating on row/col 6)
    for i in range(8, size - 8):
        val = 1.0 if i % 2 == 1 else 0.0
        r[6, i] = val
        r[i, 6] = val

    # Dark module
    r[4 * version + 9, 8] = 0.0

    # Alignment patterns
    if version >= 2 and version in _QR_ALIGNMENT_POSITIONS:
        ap = 1.0 - _ALIGNMENT_PATTERN
        for rc in _QR_ALIGNMENT_POSITIONS[version]:
            for cc in _QR_ALIGNMENT_POSITIONS[version]:
                r0, r1 = rc - 2, rc + 3
                c0, c1 = cc - 2, cc + 3
                # Skip positions that overlap finder+separator regions
                if r0 < 8 and c0 < 8:
                    continue
                if r0 < 8 and c1 > size - 8:
                    continue
                if r1 > size - 8 and c0 < 8:
                    continue
                r[r0:r1, c0:c1] = ap

    return r


def encode_qr(data, version=None, error_correction='M'):
    """Encode data as a QR module array (1=white, 0=black). Returns (modules, version)."""
    import qrcode

    if version is not None and (not isinstance(version, int) or version < 1 or version > 40):
        raise ValueError(f"QR version must be an integer 1-40, got {version!r}")
    if error_correction not in ('L', 'M', 'Q', 'H'):
        raise ValueError(f"error_correction must be one of 'L', 'M', 'Q', 'H', got {error_correction!r}")

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

    modules = np.array(qr.modules, dtype=np.float64)
    return 1.0 - modules, qr.version


# --- Signal operations ---

def upscale(x, m):
    """Repeat each element m times along each axis."""
    if x.ndim == 1:
        return np.repeat(x, m)
    return np.kron(x, np.ones((m, m)))


def downscale_sum(y, m):
    """Sum non-overlapping m-blocks along each axis (adjoint of upscale)."""
    if y.ndim == 1:
        n = len(y) // m
        return y[:n * m].reshape(n, m).sum(axis=1)
    nh, nw = y.shape[0] // m, y.shape[1] // m
    return y[:nh * m, :nw * m].reshape(nh, m, nw, m).sum(axis=(1, 3))


def _flip(x):
    if x.ndim == 1:
        return x[::-1]
    return x[::-1, ::-1]


# --- Numerically stable helpers for the dual objectives ---

def _log_partition(z, r):
    """log(1 - r + r*exp(z)), stable. This is the KL conjugate per variable (eq 13)."""
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
    """r*exp(z) / (1-r+r*exp(z)), stable. Gradient of _log_partition; gives x_hat (eq 15)."""
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
    mask = w > 0
    zm, wm = z[mask], w[mask]
    c = np.max(zm)
    return c + np.log(np.sum(wm * np.exp(zm - c)))


def _softmax_w(z, w):
    out = np.zeros_like(z)
    mask = w > 0
    zm, wm = z[mask], w[mask]
    c = np.max(zm)
    e = wm * np.exp(zm - c)
    out[mask] = e / np.sum(e)
    return out


# --- Optimization defaults ---
# These follow the paper's recommendations (Rioux et al., TPAMI 2021).

# Fidelity weight for image estimation (eq 9). Large values enforce close
# agreement between the reconstructed image and the observation; the paper
# uses 1e6 as the baseline for the dual formulation's quadratic penalty.
DEFAULT_ALPHA = 1e6

# Fidelity weight for kernel estimation (eq 16). Same role as alpha but
# for the blur kernel sub-problem. Balanced with alpha so neither
# sub-problem dominates the alternating optimization.
DEFAULT_BETA = 1e6

# Gradient tolerance for L-BFGS convergence. 1e-10 is tight enough to
# avoid premature termination on the smooth dual objectives while still
# converging in reasonable time for typical barcode/QR sizes.
DEFAULT_GTOL = 1e-10

# Maximum L-BFGS iterations per sub-problem solve. 500 is generous;
# most problems converge well before this limit.
DEFAULT_MAXITER = 500

# Alternating iterations per kernel width. The paper sweeps kernel widths
# coarse-to-fine; 5 inner iterations gives each width enough passes to
# settle before moving on.
DEFAULT_INNER_ITERS = 5


# --- Image estimation (Section 2.1) ---

def estimate_image(b, c, r, m, alpha, gtol=DEFAULT_GTOL, maxiter=DEFAULT_MAXITER):
    """Solve the image dual (eq 9) via L-BFGS, recover x_hat via eq 15."""
    if m < 1:
        raise ValueError(f"Pixels per module m must be >= 1, got {m}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
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


# --- Kernel estimation (Section 2.2) ---

def _xt_lam(lam, signal_flip, Nm_shape, kernel_shape):
    """X^T lambda: adjoint of convolution-with-signal, maps signal-space -> kernel-space."""
    full = fftconvolve(lam, signal_flip, mode='full')

    if lam.ndim == 1:
        kw = kernel_shape if isinstance(kernel_shape, int) else kernel_shape[0]
        Nm = Nm_shape if isinstance(Nm_shape, int) else Nm_shape[0]
        offset = (kw - 1) // 2
        start = Nm - 1 - offset
        return full[start : start + kw].copy()

    kh, kw = kernel_shape if not isinstance(kernel_shape, int) else (kernel_shape, kernel_shape)
    Nmh, Nmw = Nm_shape if not isinstance(Nm_shape, int) else (Nm_shape, Nm_shape)
    start_h = Nmh - 1 - (kh - 1) // 2
    start_w = Nmw - 1 - (kw - 1) // 2
    return full[start_h : start_h + kh, start_w : start_w + kw].copy()


def estimate_kernel(b, x_hat, kernel_shape, m, beta, gtol=DEFAULT_GTOL, maxiter=DEFAULT_MAXITER):
    """Solve the kernel dual (eq 16) via L-BFGS, recover c_hat via weighted softmax (eq 17)."""
    if m < 1:
        raise ValueError(f"Pixels per module m must be >= 1, got {m}")
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
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


# --- Algorithm 1: blind deblurring loop ---

UPCA_QUIET_ZONE = 9
QR_QUIET_ZONE = 4


def prepare_signal_1d(x, m, kernel, noise_var=0.0):
    """Invert, pad with quiet zone, upscale, and blur a 1D barcode."""
    qz = UPCA_QUIET_ZONE
    r = upca_symbolic_prior()
    x_inv, r_inv = 1.0 - x, 1.0 - r
    x_padded = np.concatenate([np.zeros(qz), x_inv, np.zeros(qz)])
    r_padded = np.concatenate([np.zeros(qz), r_inv, np.zeros(qz)])
    return blur_signal(upscale(x_padded, m), kernel, noise_var), r_padded


def prepare_signal_2d(x, m, kernel, noise_var=0.0, version=None):
    """Invert, pad with quiet zone, upscale, and blur a 2D QR code."""
    qz = QR_QUIET_ZONE
    if version is None:
        version = (x.shape[0] - 17) // 4
    r = qr_symbolic_prior(version)
    x_inv, r_inv = 1.0 - x, 1.0 - r
    x_padded = np.pad(x_inv, qz, mode='constant', constant_values=0)
    r_padded = np.pad(r_inv, qz, mode='constant', constant_values=0)
    return blur_signal(upscale(x_padded, m), kernel, noise_var), r_padded


def entropic_blind_deblur(b, r, m, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA,
                          max_kernel_width=None, inner_iters=DEFAULT_INNER_ITERS,
                          check_fn=None, extract_fn=None,
                          snapshots=None, verbose=True):
    """Alternating kernel/image estimation with coarse-to-fine kernel widths.

    check_fn(x_modules) should return decoded string or None.
    extract_fn(x_thresh) should strip quiet zone and invert back to image convention.
    snapshots: if a list is passed, intermediate (x_modules, c_hat, kw, iter) tuples
               are appended after each iteration for visualization.
    """
    b = np.asarray(b, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    if b.ndim not in (1, 2):
        raise ValueError(f"b must be 1D or 2D, got shape {b.shape}")
    if b.ndim != r.ndim:
        raise ValueError(f"b and r must have the same number of dimensions: "
                         f"b is {b.ndim}D, r is {r.ndim}D")
    if m < 1:
        raise ValueError(f"Pixels per module m must be >= 1, got {m}")
    if inner_iters < 1:
        raise ValueError(f"inner_iters must be >= 1, got {inner_iters}")

    is_2d = b.ndim == 2

    if max_kernel_width is None:
        max_kernel_width = min(b.shape) // 2

    if extract_fn is None:
        qz = UPCA_QUIET_ZONE
        def extract_fn(x_thresh):
            return 1.0 - x_thresh[qz : qz + UPCA_N]

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
            x_modules = extract_fn(x_thresh)

            if snapshots is not None:
                snapshots.append((x_modules.copy(), c_hat.copy(), kw, j + 1))

            if check_fn is not None:
                result = check_fn(x_modules)
                if result:
                    if verbose:
                        print(f"Decoded at kernel_width={kw}, iter={j+1}: {result}")
                    return x_modules, c_hat, True

        if verbose:
            print(f"kernel_width={kw}: not decoded")

    if verbose:
        print("Terminated without decoding")
    return extract_fn((x_hat > 0.5).astype(np.float64)), c_hat, False


# --- Readability checks ---

def make_check_fn(m):
    """Barcode checker: renders at scale m, runs pyzbar."""
    try:
        from pyzbar.pyzbar import decode
    except ImportError:
        print("pyzbar not available; readability checking disabled")
        return None

    def check(x):
        qz = np.ones(9 * m)
        row = np.concatenate([qz, upscale(x, m), qz])
        img = np.tile((row * 255).astype(np.uint8).reshape(1, -1), (50, 1))
        detected = decode(img)
        return detected[0].data.decode() if detected else None

    return check


def make_qr_check_fn(m):
    """QR checker: pads with quiet zone, upscales, runs pyzbar."""
    try:
        from pyzbar.pyzbar import decode
    except ImportError:
        print("pyzbar not available; readability checking disabled")
        return None

    def check(x):
        padded = np.pad(x, QR_QUIET_ZONE, mode='constant', constant_values=1.0)
        img = (upscale(padded, m) * 255).astype(np.uint8)
        detected = decode(img)
        return detected[0].data.decode() if detected else None

    return check


def make_extract_fn_1d(quiet_zone, n_modules):
    def extract(x_thresh):
        return 1.0 - x_thresh[quiet_zone : quiet_zone + n_modules]
    return extract


def make_extract_fn_2d(quiet_zone, size):
    def extract(x_thresh):
        return 1.0 - x_thresh[quiet_zone : quiet_zone + size,
                               quiet_zone : quiet_zone + size]
    return extract


# --- Kernels ---

def gaussian_kernel_1d(width, sigma):
    if width < 1 or width % 2 == 0:
        raise ValueError(f"Kernel width must be a positive odd integer, got {width}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    t = np.arange(width) - width // 2
    k = np.exp(-t**2 / (2 * sigma**2))
    return k / k.sum()


def gaussian_kernel_2d(width, sigma):
    if width < 1 or width % 2 == 0:
        raise ValueError(f"Kernel width must be a positive odd integer, got {width}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    t = np.arange(width) - width // 2
    g = np.exp(-t**2 / (2 * sigma**2))
    k = np.outer(g, g)
    return k / k.sum()


def box_kernel_1d(width):
    return np.ones(width) / width


def box_kernel_2d(width):
    return np.ones((width, width)) / (width * width)


def motion_kernel(width, angle_deg=0):
    """1-pixel-wide linear motion blur at the given angle."""
    k = np.zeros((width, width))
    center = width // 2
    cos_a = np.cos(np.deg2rad(angle_deg))
    sin_a = np.sin(np.deg2rad(angle_deg))

    for i in range(width):
        t = i - center
        xi, yi = int(round(center + t * cos_a)), int(round(center + t * sin_a))
        if 0 <= xi < width and 0 <= yi < width:
            k[yi, xi] = 1.0

    return k / k.sum()


def blur_signal(signal, kernel, noise_var=0.0):
    b = fftconvolve(signal, kernel, mode='same')
    if noise_var > 0:
        b += np.random.normal(0, np.sqrt(noise_var), b.shape)
        b = np.clip(b, 0, 1)
    return b


# --- Visualization helpers ---

def _barcode_img(x, m, height=60):
    """Render a 1D barcode module array as a 2D grayscale image."""
    row = upscale(x, m)
    return np.tile((row * 255).astype(np.uint8).reshape(1, -1), (height, 1))


def _qr_img(x, m):
    """Render a 2D QR module array as a grayscale image."""
    return (upscale(x, m) * 255).astype(np.uint8)


def _pick_snapshots(snapshots, n=4):
    """Pick n evenly-spaced snapshots from the list (always including last)."""
    if len(snapshots) <= n:
        return snapshots
    indices = np.linspace(0, len(snapshots) - 1, n, dtype=int)
    return [snapshots[i] for i in indices]


# --- Demos ---

def demo(blur_width=21, sigma=1.0, m=3, noise_var=0.0,
         alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, max_kernel_width=15, show=True):
    """Blind-deblur a synthetic UPC-A barcode with visualization."""
    import random
    import matplotlib.pyplot as plt

    digits = ''.join([str(random.randint(0, 9)) for _ in range(11)])
    digits += compute_check_digit(digits)
    print(f"Original barcode: {digits}")

    x = encode_upca(digits)
    kernel = gaussian_kernel_1d(blur_width, sigma)
    b, r_inv = prepare_signal_1d(x, m, kernel, noise_var)
    print(f"Gaussian blur: width={blur_width}, sigma={sigma:.1f}, m={m}")

    check = make_check_fn(m)
    extract = make_extract_fn_1d(UPCA_QUIET_ZONE, UPCA_N)

    # First pass: collect all snapshots (no early stopping)
    snaps = []
    entropic_blind_deblur(
        b, r_inv, m, alpha=alpha, beta=beta,
        max_kernel_width=max_kernel_width,
        check_fn=None, extract_fn=extract,
        snapshots=snaps, verbose=False
    )

    # Find which snapshot first decoded
    decoded_idx = None
    for i, (x_snap, _, kw, it) in enumerate(snaps):
        if check and check(x_snap):
            decoded_idx = i
            break

    # Pick snapshots to show — always include first, decoded, and last
    if decoded_idx is not None:
        # Show progression up to and including the decode point
        pool = snaps[:decoded_idx + 1]
        picked = _pick_snapshots(pool, n=4)
        final_snap = snaps[decoded_idx]
        x_hat = final_snap[0]
        success = True
        kw_d, it_d = final_snap[2], final_snap[3]
        print(f"Decoded at kernel_width={kw_d}, iter={it_d}")
    else:
        picked = _pick_snapshots(snaps, n=4)
        x_hat = snaps[-1][0] if snaps else extract(r_inv)
        success = False
        print("Terminated without decoding")

    accuracy = np.mean(x_hat == x) * 100
    print(f"{'SUCCESS' if success else 'FAILED'} — {accuracy:.1f}% module accuracy")

    if not show:
        return x, x_hat, b, kernel

    n_snaps = len(picked)
    n_cols = 2 + n_snaps

    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 4),
                             gridspec_kw={'height_ratios': [3, 1]})

    bar_h = 80

    axes[0, 0].imshow(_barcode_img(x, m, bar_h), cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original', fontsize=9)
    axes[1, 0].plot(upscale(x, m), 'k', linewidth=0.5)
    axes[1, 0].set_ylim(-0.1, 1.1)
    axes[1, 0].set_title('Signal', fontsize=8)

    blurred_img = np.tile((np.clip(b, 0, 1) * 255).astype(np.uint8).reshape(1, -1), (bar_h, 1))
    axes[0, 1].imshow(blurred_img, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title(f'Blurred (w={blur_width})', fontsize=9)
    axes[1, 1].plot(b, 'b', linewidth=0.5)
    axes[1, 1].set_ylim(-0.1, 1.1)
    axes[1, 1].set_title('Blurred signal', fontsize=8)

    for col, (x_snap, c_snap, kw, it) in enumerate(picked, start=2):
        acc = np.mean(x_snap == x) * 100
        is_decoded = check and check(x_snap)
        label = f'kw={kw} i={it}\n{acc:.0f}%'
        if is_decoded:
            label += ' *'
        axes[0, col].imshow(_barcode_img(x_snap, m, bar_h), cmap='gray', vmin=0, vmax=255)
        axes[0, col].set_title(label, fontsize=9,
                               color='green' if is_decoded else 'black',
                               fontweight='bold' if is_decoded else 'normal')
        axes[1, col].bar(range(len(c_snap)), c_snap, color='steelblue', width=0.8)
        axes[1, col].set_title(f'Kernel ({kw})', fontsize=8)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f'UPC-A Blind Deblurring — {digits}', fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig('demo_barcode.png', dpi=150, bbox_inches='tight')
    print("Saved demo_barcode.png")
    plt.show()

    return x, x_hat, b, kernel


def demo_qr(data="HELLO WORLD", blur_width=5, sigma=1.0, m=5,
            noise_var=0.0, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA,
            error_correction='M', version=None, max_kernel_width=11, show=True):
    """Blind-deblur a synthetic QR code with visualization."""
    import matplotlib.pyplot as plt

    x, ver = encode_qr(data, version=version, error_correction=error_correction)
    size = qr_size(ver)
    print(f"QR data={data!r}, version {ver} ({size}x{size})")

    kernel = gaussian_kernel_2d(blur_width, sigma)
    b, r_inv = prepare_signal_2d(x, m, kernel, noise_var, version=ver)
    print(f"Gaussian blur: width={blur_width}, sigma={sigma:.1f}, signal {b.shape}")

    check = make_qr_check_fn(m)
    extract = make_extract_fn_2d(QR_QUIET_ZONE, size)

    # Collect all snapshots without early stopping
    snaps = []
    entropic_blind_deblur(
        b, r_inv, m, alpha=alpha, beta=beta,
        max_kernel_width=max_kernel_width,
        check_fn=None, extract_fn=extract,
        snapshots=snaps, verbose=False
    )

    decoded_idx = None
    for i, (x_snap, _, kw, it) in enumerate(snaps):
        if check and check(x_snap):
            decoded_idx = i
            break

    if decoded_idx is not None:
        pool = snaps[:decoded_idx + 1]
        picked = _pick_snapshots(pool, n=4)
        x_hat = snaps[decoded_idx][0]
        success = True
        kw_d, it_d = snaps[decoded_idx][2], snaps[decoded_idx][3]
        print(f"Decoded at kernel_width={kw_d}, iter={it_d}")
    else:
        picked = _pick_snapshots(snaps, n=4)
        x_hat = snaps[-1][0] if snaps else extract(r_inv)
        success = False
        print("Terminated without decoding")

    accuracy = np.mean(x_hat == x) * 100
    print(f"{'SUCCESS' if success else 'FAILED'} — {accuracy:.1f}% module accuracy")

    if not show:
        return x, x_hat, b, kernel

    n_snaps = len(picked)
    n_cols = 2 + n_snaps

    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6),
                             gridspec_kw={'height_ratios': [3, 1]})

    axes[0, 0].imshow(_qr_img(x, m), cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    axes[0, 0].set_title('Original', fontsize=9)
    axes[1, 0].axis('off')

    blurred_vis = (np.clip(b, 0, 1) * 255).astype(np.uint8)
    axes[0, 1].imshow(blurred_vis, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    axes[0, 1].set_title(f'Blurred (w={blur_width})', fontsize=9)
    axes[1, 1].imshow(kernel, cmap='hot', interpolation='nearest')
    axes[1, 1].set_title('True kernel', fontsize=8)

    for col, (x_snap, c_snap, kw, it) in enumerate(picked, start=2):
        acc = np.mean(x_snap == x) * 100
        is_decoded = check and check(x_snap)
        label = f'kw={kw} i={it}\n{acc:.0f}%'
        if is_decoded:
            label += ' *'
        axes[0, col].imshow(_qr_img(x_snap, m), cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        axes[0, col].set_title(label, fontsize=9,
                               color='green' if is_decoded else 'black',
                               fontweight='bold' if is_decoded else 'normal')
        axes[1, col].imshow(c_snap, cmap='hot', interpolation='nearest')
        axes[1, col].set_title(f'Kernel ({kw}x{kw})', fontsize=8)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f'QR Blind Deblurring — {data!r}', fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig('demo_qr.png', dpi=150, bbox_inches='tight')
    print("Saved demo_qr.png")
    plt.show()

    return x, x_hat, b, kernel


def demo_recovery():
    """Compare full demo (exhaustive) vs recovery mode (fast fallback)."""
    import random
    import time

    digits = ''.join([str(random.randint(0, 9)) for _ in range(11)])
    digits += compute_check_digit(digits)

    x = encode_upca(digits)
    kernel = gaussian_kernel_1d(21, 1.0)
    b, r_inv = prepare_signal_1d(x, 3, kernel)

    print(f"\n{'='*60}")
    print(f"Recovery-mode comparison — UPC-A {digits}")
    print(f"{'='*60}")

    # Full (demo) mode
    check = make_check_fn(3)
    extract = make_extract_fn_1d(UPCA_QUIET_ZONE, UPCA_N)

    t0 = time.perf_counter()
    _, _, success_full = entropic_blind_deblur(
        b, r_inv, m=3, max_kernel_width=15,
        check_fn=check, extract_fn=extract, verbose=False,
    )
    t_full = time.perf_counter() - t0
    print(f"Full mode:     {'OK' if success_full else 'FAIL'} in {t_full:.2f}s")

    # Recovery mode (fast)
    try:
        from .deblur_recovery import recover_barcode
    except ImportError:
        from deblur_recovery import recover_barcode
    t0 = time.perf_counter()
    result = recover_barcode(b, m=3, max_kernel_width=15)
    t_fast = time.perf_counter() - t0
    print(f"Recovery mode: {'OK' if result.success else 'FAIL'} in {t_fast:.2f}s "
          f"({result.iterations} iters, kw={result.kernel_width})")

    if t_full > 0:
        print(f"Speedup: {t_full / t_fast:.1f}x")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Blind deblurring of barcodes')
    parser.add_argument('mode', nargs='?', default='barcode',
                        choices=['barcode', 'qr', 'both', 'recovery'],
                        help='Which demo to run (default: barcode)')
    parser.add_argument('--blur-width', type=int, default=None,
                        help='Blur kernel width (default: 21 for barcode, 5 for QR)')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Gaussian sigma (default: 1.0)')
    parser.add_argument('--m', type=int, default=None,
                        help='Pixels per module (default: 3 for barcode, 5 for QR)')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='Noise variance (default: 0.0)')
    parser.add_argument('--data', type=str, default='HELLO WORLD',
                        help='QR data string (default: "HELLO WORLD")')
    parser.add_argument('--no-show', action='store_true',
                        help='Skip matplotlib display (still saves PNG)')

    args = parser.parse_args()

    if args.mode in ('barcode', 'both'):
        demo(
            blur_width=args.blur_width or 21,
            sigma=args.sigma,
            m=args.m or 3,
            noise_var=args.noise,
            show=not args.no_show,
        )

    if args.mode in ('qr', 'both'):
        demo_qr(
            data=args.data,
            blur_width=args.blur_width or 5,
            sigma=args.sigma,
            m=args.m or 5,
            noise_var=args.noise,
            show=not args.no_show,
        )

    if args.mode == 'recovery':
        demo_recovery()
