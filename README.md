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
pip install .
```

pyzbar needs the zbar system library (`brew install zbar` on macOS, `apt install libzbar0` on Debian/Ubuntu).

For the visualization demos: `pip install ".[demo]"`

## Usage

### Drop-in recovery mode

Use `recover_barcode` / `recover_qr` as a fallback when your barcode scanner chokes on a blurry image. These use relaxed solver parameters for speed and bail out as soon as decoding succeeds.

```python
from deblurrinator import recover_barcode, recover_qr

result = recover_barcode(blurred_signal, m=3)
if result.success:
    print(result.data)       # decoded string
    print(result.modules)    # recovered binary module array

result = recover_qr(blurred_image, m=5, version=1)
if result.success:
    print(result.data)
```

The result object also has `kernel` (estimated blur kernel), `iterations`, and `kernel_width`.

### Full algorithm

For more control (or heavier blur), use the core algorithm directly. This runs the full alternating optimization with tighter tolerances.

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

### Demos

```python
from deblurrinator import demo, demo_qr

demo(blur_width=21, sigma=1.0)        # 1D UPC-A
demo_qr(data="HELLO WORLD", blur_width=5, sigma=1.0, m=5)  # QR
```

Or: `python -m deblurrinator.entropic_deblur`

## Notes

- `m` is pixels per module — higher means more signal to work with but slower solves. Recovery mode defaults to 3, full mode to 5.
- **macOS**: pyzbar may need `DYLD_LIBRARY_PATH=/opt/homebrew/lib` to find zbar.
- **QR is harder**: works well for moderate blur (kernel width ~5 with m=5). Heavier blur may need parameter tuning or higher error correction.
- The inverted convention (bars=1, spaces=0) ensures quiet zones map to zero, matching the zero-padding implicit in `fftconvolve`.
