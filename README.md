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

## Usage

```bash
pip install -r requirements.txt
brew install zbar  # needed by pyzbar
```

### 1D barcode (UPC-A)

```python
from src.entropic_deblur import demo
x, x_hat, b, c_hat, kernel = demo(blur_width=21, sigma=1.0)
```

### QR code

```python
from src.entropic_deblur import demo_qr
x, x_hat, b, c_hat, kernel = demo_qr(data="HELLO WORLD", blur_width=5, sigma=1.0, m=5)
```

### Custom usage

```python
from src.entropic_deblur import (
    encode_upca, upca_symbolic_prior, prepare_signal_1d,
    entropic_blind_deblur, make_check_fn, make_extract_fn_1d,
    gaussian_kernel_1d, UPCA_QUIET_ZONE, UPCA_N
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

| Parameter | Default | Notes |
|-----------|---------|-------|
| `m` | 5 | Pixels per module. Higher = better accuracy but slower. |
| `alpha` | 1e6 | Image estimation fidelity weight. |
| `beta` | 1e6 | Kernel estimation fidelity weight. |
| `inner_iters` | 5 | Alternating iterations per kernel width. |
| `max_kernel_width` | auto | Largest kernel to try. Defaults to half the signal length. |

## Notes

- **macOS**: pyzbar may need `DYLD_LIBRARY_PATH=/opt/homebrew/lib` to find zbar.
- **2D is harder**: QR blind deblurring works well for moderate blur (kernel width up to ~9 pixels with m=5). Heavier blur may need parameter tuning or higher error correction levels.
- The inverted convention (bars=1, spaces=0) ensures quiet zones map to zero, matching the zero-padding implicit in `fftconvolve`.

## Requirements

- Python 3.8+
- numpy, scipy, opencv-python, python-barcode, pyzbar, qrcode
- zbar (system library, for pyzbar)
