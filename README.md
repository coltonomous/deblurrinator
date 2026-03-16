# deblurrinator

Blind deblurring of 1D barcodes (UPC-A) and 2D barcodes (QR codes) using entropy-regularized optimization. Recovers readable barcodes from blurry images where standard scanners fail.

Based on:

> H. Rioux, N. Scarvelis, R. Choksi, T. Hoheisel, and P. Marechal,
> **"Blind Deblurring of Barcodes via Kullback-Leibler Divergence,"**
> *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 43, no. 1, pp. 77–88, Jan. 2021.
> [[PDF]](https://www.math.mcgill.ca/rchoksi/pub/KL.pdf)

## How it works

A barcode is modeled as N independent Bernoulli random variables — each module is either black or white. Given a blurry observation, we want the probability distribution over all 2^N possible barcodes that best explains what we see, regularized by KL divergence from a symbolic prior that encodes the barcode's known structural constraints (guard patterns, timing patterns, etc.).

Directly optimizing over 2^N distributions is intractable. Fenchel-Rockafellar duality reduces this to a dual problem with only Nm variables (pixels in the blurred signal), which L-BFGS handles easily. The algorithm alternates between estimating the blur kernel and estimating the barcode image, sweeping kernel widths from small to large until the result decodes.

## Performance

Benchmarked across 270 synthetic trials (6 blur widths × 3 noise levels × 3 kernel types × 5 trials each):

| Kernel type | Blur width | Noise = 0 | Noise = 0.005 | Noise = 0.01 |
|-------------|-----------|-----------|---------------|--------------|
| Gaussian    | 5–25      | **100%**  | **100%**      | **100%**     |
| Box         | 5         | **100%**  | **100%**      | **100%**     |
| Box         | 9         | 60%       | —             | —            |
| Motion      | 5         | **100%**  | **100%**      | 80%          |
| Motion      | 9         | 60%       | —             | —            |

Gaussian blur decodes perfectly at all tested widths and noise levels. Box and motion blur work reliably at moderate widths (≤5) and partially at width 9. Full benchmark data in `benchmark_results_1d.json`.

## Installation

```bash
pip install .
```

**System dependency:** pyzbar needs the zbar library:
- macOS: `brew install zbar`
- Debian/Ubuntu: `apt install libzbar0`
- macOS may also need: `export DYLD_LIBRARY_PATH=/opt/homebrew/lib`

For visualization demos: `pip install ".[demo]"`

## Quick start

### Deblur a photograph

```python
from deblurrinator import deblur_from_image

result = deblur_from_image("blurry_barcode.jpg", barcode_type='1d', m=3)
if result.success:
    print(result.data)
```

For QR codes:

```python
result = deblur_from_image("blurry_qr.jpg", barcode_type='2d', m=5, version=2)
```

### Recovery mode (from signal data)

Use `recover_barcode` / `recover_qr` when you already have a blurred signal as a numpy array:

```python
from deblurrinator import recover_barcode, recover_qr

result = recover_barcode(blurred_signal, m=3)
if result.success:
    print(result.data)       # decoded string
    print(result.modules)    # recovered binary module array

result = recover_qr(blurred_image, m=5, version=1)
```

### Full algorithm

For heavier blur or more control, use the core algorithm with tighter tolerances:

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

### Live camera / video

Process a live camera feed or video file with warm-started optimization for faster frame-to-frame decoding:

```bash
python -m deblurrinator.entropic_deblur camera --source 0 --m 3 --debug
python -m deblurrinator.entropic_deblur video --source scan.mp4 --barcode-type 1d --m 3
```

Or from Python:

```python
from deblurrinator import live_camera, process_video

live_camera(source=0, barcode_type='1d', m=3, show_debug=True)

results = process_video("scan.mp4", barcode_type='1d', m=3)
for r in results:
    if r.result.success:
        print(f"Frame {r.frame_id}: {r.result.data} ({r.processing_time_ms:.0f}ms)")
```

The streaming pipeline reuses L-BFGS dual variables between frames (warm-starting), giving ~8x speedup on consecutive similar frames. A sharpness filter skips the blurriest ~30% of frames automatically.

### Benchmarking

Run a parametric sweep across blur widths, noise levels, and kernel types:

```python
from deblurrinator import run_benchmark, BenchmarkConfig, print_results_table, plot_results

config = BenchmarkConfig(
    blur_widths=[5, 9, 13, 17, 21],
    noise_levels=[0.0, 0.005, 0.01],
    kernel_types=['gaussian', 'box', 'motion'],
    n_trials=5,
)
results = run_benchmark(config)
print_results_table(results)
plot_results(results, save_path='bench')
```

## CLI

```bash
python -m deblurrinator.entropic_deblur              # 1D barcode demo (Gaussian blur)
python -m deblurrinator.entropic_deblur qr            # QR code demo
python -m deblurrinator.entropic_deblur both          # both demos
python -m deblurrinator.entropic_deblur recovery      # recovery-mode speed comparison
python -m deblurrinator.entropic_deblur kernels       # side-by-side kernel type comparison

# Kernel type and parameters
python -m deblurrinator.entropic_deblur --kernel motion --blur-width 21
python -m deblurrinator.entropic_deblur qr --kernel box --blur-width 5
python -m deblurrinator.entropic_deblur --blur-width 31 --m 5 --noise 0.01

# Camera and video
python -m deblurrinator.entropic_deblur camera --source 0 --m 3 --debug
python -m deblurrinator.entropic_deblur video --source clip.mp4
```

## Project structure

```
deblurrinator/
  entropic_deblur.py    Core algorithm, encoding, kernels, demos, CLI
  deblur_recovery.py    Fast recovery mode with tuned parameters
  image_input.py        Real image pipeline (load, ROI detect, extract)
  streaming.py          Video/camera mode with warm-started optimization
  benchmark.py          Parametric benchmark suite
tests/
  test_deblurrinator.py 44 tests (encoding, signal ops, recovery, validation)
```

## Tests

```bash
pip install pytest
DYLD_LIBRARY_PATH=/opt/homebrew/lib pytest tests/ -v
```

## Notes

- `m` is pixels per module — higher means more signal but slower solves. Recovery mode defaults to 3.
- The algorithm works best on Gaussian blur. Box and motion blur are harder due to sharp kernel edges that the softmax parameterization biases against.
- QR codes are harder than 1D barcodes — moderate blur (kernel width ~5 with m=5) works well; heavier blur may need the full algorithm or higher error correction.
- Live camera mode: press `q` to quit, `r` to reset warm-start state, `d` to toggle debug overlay.
