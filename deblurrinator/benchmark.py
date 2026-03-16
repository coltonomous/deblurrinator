"""Benchmark suite for entropic barcode deblurring.

Generates synthetic barcodes at varying blur widths, noise levels, and kernel
types, then measures decode success rate, module accuracy, and runtime.

    from deblurrinator.benchmark import run_benchmark, BenchmarkConfig
    results = run_benchmark(BenchmarkConfig())
    print_results_table(results)
"""

import random
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .entropic_deblur import (
    UPCA_N,
    UPCA_QUIET_ZONE,
    QR_QUIET_ZONE,
    compute_check_digit,
    encode_upca,
    encode_qr,
    qr_size,
    prepare_signal_1d,
    prepare_signal_2d,
    gaussian_kernel_1d,
    gaussian_kernel_2d,
    box_kernel_1d,
    box_kernel_2d,
    motion_kernel,
    make_check_fn,
    make_qr_check_fn,
    make_extract_fn_1d,
    make_extract_fn_2d,
    entropic_blind_deblur,
)
from .deblur_recovery import recover_barcode, recover_qr


@dataclass
class BenchmarkConfig:
    """Parameters for a benchmark sweep."""
    blur_widths: list = field(default_factory=lambda: [5, 9, 13, 17, 21, 25])
    noise_levels: list = field(default_factory=lambda: [0.0, 0.001, 0.005, 0.01])
    kernel_types: list = field(default_factory=lambda: ['gaussian', 'box', 'motion'])
    n_trials: int = 10
    barcode_type: str = '1d'  # '1d', '2d', or 'both'
    m: int = 3
    sigma: float = 1.0
    motion_angle: float = 0.0
    use_recovery_mode: bool = True
    max_kernel_width: int = 15


@dataclass
class BenchmarkResult:
    """Result from a single benchmark trial."""
    barcode_type: str
    kernel_type: str
    blur_width: int
    noise_level: float
    success: bool
    module_accuracy: float
    runtime_seconds: float
    decoded_data: Optional[str] = None


def _make_kernel_1d(kernel_type, width, sigma):
    if kernel_type == 'gaussian':
        return gaussian_kernel_1d(width, sigma)
    elif kernel_type in ('box', 'motion'):
        return box_kernel_1d(width)
    raise ValueError(f"Unknown kernel type {kernel_type!r}")


def _make_kernel_2d(kernel_type, width, sigma, angle=0.0):
    if kernel_type == 'gaussian':
        return gaussian_kernel_2d(width, sigma)
    elif kernel_type == 'box':
        return box_kernel_2d(width)
    elif kernel_type == 'motion':
        return motion_kernel(width, angle_deg=angle)
    raise ValueError(f"Unknown kernel type {kernel_type!r}")


def _random_digits():
    digits = ''.join([str(random.randint(0, 9)) for _ in range(11)])
    return digits + compute_check_digit(digits)


def _run_1d_trial(config, kernel_type, blur_width, noise_level):
    digits = _random_digits()
    x = encode_upca(digits)
    kernel = _make_kernel_1d(kernel_type, blur_width, config.sigma)
    b, r_inv = prepare_signal_1d(x, config.m, kernel, noise_level)

    t0 = time.perf_counter()
    if config.use_recovery_mode:
        result = recover_barcode(b, m=config.m, max_kernel_width=config.max_kernel_width)
        success = result.success
        decoded = result.data
        # Get modules for accuracy
        x_hat = result.modules
    else:
        check = make_check_fn(config.m)
        extract = make_extract_fn_1d(UPCA_QUIET_ZONE, UPCA_N)
        x_hat, _, success = entropic_blind_deblur(
            b, r_inv, config.m, max_kernel_width=config.max_kernel_width,
            check_fn=check, extract_fn=extract, verbose=False,
        )
        decoded = digits if success else None
    elapsed = time.perf_counter() - t0

    accuracy = np.mean(x_hat == x) * 100 if x_hat is not None else 0.0
    return BenchmarkResult(
        barcode_type='1d', kernel_type=kernel_type,
        blur_width=blur_width, noise_level=noise_level,
        success=success, module_accuracy=accuracy,
        runtime_seconds=elapsed, decoded_data=decoded,
    )


def _run_2d_trial(config, kernel_type, blur_width, noise_level):
    # Short random string for QR
    data = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=8))
    x, ver = encode_qr(data)
    size = qr_size(ver)
    kernel = _make_kernel_2d(kernel_type, blur_width, config.sigma, config.motion_angle)
    b, r_inv = prepare_signal_2d(x, config.m, kernel, noise_level, version=ver)

    t0 = time.perf_counter()
    if config.use_recovery_mode:
        result = recover_qr(b, m=config.m, version=ver, max_kernel_width=config.max_kernel_width)
        success = result.success
        decoded = result.data
        x_hat = result.modules
    else:
        check = make_qr_check_fn(config.m)
        extract = make_extract_fn_2d(QR_QUIET_ZONE, size)
        x_hat, _, success = entropic_blind_deblur(
            b, r_inv, config.m, max_kernel_width=config.max_kernel_width,
            check_fn=check, extract_fn=extract, verbose=False,
        )
        decoded = data if success else None
    elapsed = time.perf_counter() - t0

    accuracy = np.mean(x_hat == x) * 100 if x_hat is not None else 0.0
    return BenchmarkResult(
        barcode_type='2d', kernel_type=kernel_type,
        blur_width=blur_width, noise_level=noise_level,
        success=success, module_accuracy=accuracy,
        runtime_seconds=elapsed, decoded_data=decoded,
    )


def run_benchmark(config=None):
    """Run a full benchmark sweep. Returns a list of BenchmarkResult."""
    if config is None:
        config = BenchmarkConfig()

    results = []
    run_1d = config.barcode_type in ('1d', 'both')
    run_2d = config.barcode_type in ('2d', 'both')

    combos = [
        (kt, bw, nl)
        for kt in config.kernel_types
        for bw in config.blur_widths
        for nl in config.noise_levels
    ]
    total = len(combos) * config.n_trials * (run_1d + run_2d)
    done = 0

    for kt, bw, nl in combos:
        # Skip motion kernel for widths that would be degenerate
        for trial in range(config.n_trials):
            if run_1d:
                results.append(_run_1d_trial(config, kt, bw, nl))
                done += 1
                print(f"\r  [{done}/{total}]", end='', flush=True)
            if run_2d:
                results.append(_run_2d_trial(config, kt, bw, nl))
                done += 1
                print(f"\r  [{done}/{total}]", end='', flush=True)

    print()
    return results


def summarize_results(results):
    """Aggregate results by (barcode_type, kernel_type, blur_width, noise_level).

    Returns a dict mapping each key to {success_rate, mean_accuracy, mean_runtime, n_trials}.
    """
    groups = {}
    for r in results:
        key = (r.barcode_type, r.kernel_type, r.blur_width, r.noise_level)
        groups.setdefault(key, []).append(r)

    summary = {}
    for key, group in groups.items():
        n = len(group)
        summary[key] = {
            'success_rate': sum(r.success for r in group) / n,
            'mean_accuracy': np.mean([r.module_accuracy for r in group]),
            'mean_runtime': np.mean([r.runtime_seconds for r in group]),
            'n_trials': n,
        }
    return summary


def print_results_table(results):
    """Pretty-print benchmark results as a text table."""
    summary = summarize_results(results)

    print(f"\n{'Type':>4s} {'Kernel':>10s} {'Width':>5s} {'Noise':>7s} "
          f"{'Success':>8s} {'Accuracy':>9s} {'Time(s)':>8s} {'N':>3s}")
    print('-' * 62)

    for key in sorted(summary.keys()):
        btype, kt, bw, nl = key
        s = summary[key]
        print(f"{btype:>4s} {kt:>10s} {bw:>5d} {nl:>7.4f} "
              f"{s['success_rate']:>7.0%} {s['mean_accuracy']:>8.1f}% "
              f"{s['mean_runtime']:>7.2f}s {s['n_trials']:>3d}")


def plot_results(results, save_path=None):
    """Plot success rate and accuracy vs blur width, grouped by kernel type."""
    import matplotlib.pyplot as plt

    summary = summarize_results(results)

    # Group by barcode type
    btypes = sorted(set(r.barcode_type for r in results))

    for btype in btypes:
        # Success rate vs blur width at noise=0
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        kernel_types = sorted(set(r.kernel_type for r in results))
        for kt in kernel_types:
            points = {k: v for k, v in summary.items()
                      if k[0] == btype and k[1] == kt and k[3] == 0.0}
            if not points:
                continue
            widths = sorted(set(k[2] for k in points))
            rates = [points[(btype, kt, w, 0.0)]['success_rate'] for w in widths]
            accs = [points[(btype, kt, w, 0.0)]['mean_accuracy'] for w in widths]
            ax1.plot(widths, rates, 'o-', label=kt, linewidth=1.5, markersize=4)
            ax2.plot(widths, accs, 'o-', label=kt, linewidth=1.5, markersize=4)

        ax1.set_xlabel('Blur width')
        ax1.set_ylabel('Decode success rate')
        ax1.set_title(f'{btype.upper()} — Success vs blur width (noise=0)')
        ax1.legend()
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Blur width')
        ax2.set_ylabel('Module accuracy (%)')
        ax2.set_title(f'{btype.upper()} — Accuracy vs blur width (noise=0)')
        ax2.legend()
        ax2.set_ylim(40, 105)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fname = f"{save_path}_benchmark_{btype}.png"
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            print(f"Saved {fname}")
        plt.show()
