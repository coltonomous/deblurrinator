# deblurrinator

Blind deblurring of 1D barcodes (UPC-A) and 2D barcodes (QR codes) using entropy-regularized optimization.

Implements the method from:

> H. Rioux, N. Scarvelis, R. Choksi, T. Hoheisel, and P. Marechal,
> **"Blind Deblurring of Barcodes via Kullback-Leibler Divergence,"**
> *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 43, no. 1, pp. 77–88, Jan. 2021.
> [[PDF]](https://www.math.mcgill.ca/rchoksi/pub/KL.pdf)

## How it works

A barcode is modeled as N independent Bernoulli random variables — each module is either black or white. Given a blurry observation, we want the probability distribution over all 2^N possible barcodes that best explains what we see, regularized by KL divergence from a symbolic prior that encodes the barcode's known structural constraints (guard patterns, timing patterns, etc.).

Directly optimizing over 2^N distributions is intractable. Fenchel-Rockafellar duality reduces this to a dual problem with only Nm variables (pixels in the blurred signal), which L-BFGS handles easily. The algorithm alternates between estimating the blur kernel and estimating the barcode image, sweeping kernel widths from small to large until the result decodes.

## Installation

```bash
pip install .               # core (numpy, scipy, pyzbar, qrcode)
pip install ".[demo]"       # + matplotlib for visualizations
pip install ".[cv]"         # + opencv, python-barcode for image utilities
```

**System dependency**: pyzbar requires the zbar library:
```bash
# macOS
brew install zbar

# Ubuntu/Debian
sudo apt-get install libzbar0

# Fedora
sudo dnf install zbar
```

## Quick start: recovery mode

The primary interface for integrating into an existing project. Use this as a fallback when your normal barcode scanner fails on a blurry image:

```python
from deblurrinator import recover_barcode, recover_qr

# 1D barcode — pass the blurred scanline (1D array, values in [0, 1])
result = recover_barcode(blurred_signal, m=3)
if result.success:
    print(result.data)       # decoded string
    print(result.modules)    # recovered binary module array

# QR code — pass the blurred image (2D array, values in [0, 1])
result = recover_qr(blurred_image, m=5, version=1)
if result.success:
    print(result.data)
```

Recovery mode uses relaxed optimization parameters (`gtol=1e-6`, `maxiter=200`, `inner_iters=2`) for faster convergence with early stopping — it returns as soon as the barcode decodes successfully.

### DeblurResult fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | Whether decoding succeeded |
| `data` | str or None | Decoded barcode string |
| `modules` | ndarray or None | Recovered binary module array |
| `kernel` | ndarray or None | Estimated blur kernel |
| `iterations` | int | Total alternating iterations performed |
| `kernel_width` | int | Kernel width at termination |

## Demos

Run the visual demos to see the algorithm push through heavy blur:

```python
from deblurrinator import demo, demo_qr

# 1D: recovers UPC-A from blur_width=21 Gaussian blur
x, x_hat, b, kernel = demo(blur_width=21, sigma=1.0)

# 2D: recovers QR code from blur_width=5 Gaussian blur
x, x_hat, b, kernel = demo_qr(data="HELLO WORLD", blur_width=5, sigma=1.0, m=5)
```

Or from the command line:
```bash
python -m deblurrinator.entropic_deblur
```

## Full algorithm access

For maximum control, use the core algorithm directly:

```python
from deblurrinator import (
    encode_upca, prepare_signal_1d, entropic_blind_deblur,
    make_check_fn, make_extract_fn_1d, gaussian_kernel_1d,
    UPCA_QUIET_ZONE, UPCA_N,
)

x = encode_upca("01234567890")
kernel = gaussian_kernel_1d(15, 1.0)
b, r = prepare_signal_1d(x, m=5, kernel=kernel)

x_hat, c_hat, success = entropic_blind_deblur(
    b, r, m=5,
    check_fn=make_check_fn(5),
    extract_fn=make_extract_fn_1d(UPCA_QUIET_ZONE, UPCA_N),
)
```

## Parameters

| Parameter | Recovery default | Full default | Notes |
|-----------|-----------------|--------------|-------|
| `m` | 3 | 5 | Pixels per module. Higher = better accuracy but slower. |
| `alpha` | 1e6 | 1e6 | Image estimation fidelity weight. |
| `beta` | 1e6 | 1e6 | Kernel estimation fidelity weight. |
| `inner_iters` | 2 | 5 | Alternating iterations per kernel width. |
| `max_kernel_width` | 15 | auto | Largest kernel to try. |
| `gtol` | 1e-6 | 1e-10 | L-BFGS gradient tolerance. |
| `maxiter` | 200 | 500 | L-BFGS max iterations per solve. |

## Notes

- **macOS**: pyzbar may need `DYLD_LIBRARY_PATH=/opt/homebrew/lib` to find zbar.
- **2D is harder**: QR blind deblurring works well for moderate blur (kernel width up to ~9 pixels with m=5). Heavier blur may need parameter tuning or higher error correction levels.
- The inverted convention (bars=1, spaces=0) ensures quiet zones map to zero, matching the zero-padding implicit in `fftconvolve`.

## Requirements

- Python 3.8+
- numpy, scipy, pyzbar, qrcode
- zbar (system library)
