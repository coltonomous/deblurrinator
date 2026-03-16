"""Tests for the deblurrinator package.

Known capabilities (from benchmark data, tuned recovery params):
  Gaussian blur: 100% decode at all widths 5-25, all noise levels 0-0.01
  Box blur:      100% decode at width 5 (all noise), 60% at width 9 (no noise)
  Motion blur:   100% at width 5 (noise<=0.005), 60% at width 9 (no noise)

Tests use fixed seeds for reproducibility but the algorithm is inherently
stochastic (random barcode digits), so we test against high success
thresholds rather than expecting 100%.
"""

import os
import random

import numpy as np
import pytest

# Set DYLD_LIBRARY_PATH for pyzbar if on macOS
if os.path.exists('/opt/homebrew/lib/libzbar.dylib'):
    os.environ.setdefault('DYLD_LIBRARY_PATH', '/opt/homebrew/lib')

from deblurrinator.entropic_deblur import (
    UPCA_N,
    UPCA_QUIET_ZONE,
    QR_QUIET_ZONE,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    encode_upca,
    encode_qr,
    compute_check_digit,
    upca_symbolic_prior,
    qr_symbolic_prior,
    qr_size,
    gaussian_kernel_1d,
    gaussian_kernel_2d,
    box_kernel_1d,
    motion_kernel,
    blur_signal,
    upscale,
    downscale_sum,
    prepare_signal_1d,
    prepare_signal_2d,
    estimate_image,
    estimate_kernel,
    entropic_blind_deblur,
    make_check_fn,
    make_extract_fn_1d,
    make_extract_fn_2d,
)
from deblurrinator.deblur_recovery import (
    recover_barcode,
    recover_qr,
    DeblurResult,
)
from deblurrinator.image_input import load_image, extract_barcode_scanline
from deblurrinator.streaming import FrameState, SharpnessFilter

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

_requires_cv2 = pytest.mark.skipif(not _HAS_CV2, reason="opencv-python not available")


# ---------------------------------------------------------------------------
# Encoding tests
# ---------------------------------------------------------------------------

class TestEncoding:
    def test_upca_length(self):
        x = encode_upca('01234567890')
        assert x.shape == (UPCA_N,)
        assert set(np.unique(x)) <= {0.0, 1.0}

    def test_upca_with_check_digit(self):
        digits = '01234567890'
        check = compute_check_digit(digits)
        x1 = encode_upca(digits)
        x2 = encode_upca(digits + check)
        np.testing.assert_array_equal(x1, x2)

    def test_upca_invalid_input(self):
        with pytest.raises(ValueError, match="numeric"):
            encode_upca('hello')
        with pytest.raises(ValueError, match="11 or 12"):
            encode_upca('123')

    def test_encode_qr_returns_binary(self):
        x, ver = encode_qr("TEST")
        assert x.ndim == 2
        assert x.shape[0] == x.shape[1]
        assert set(np.unique(x)) <= {0.0, 1.0}
        assert ver >= 1

    def test_encode_qr_invalid_version(self):
        with pytest.raises(ValueError, match="1-40"):
            encode_qr("TEST", version=0)
        with pytest.raises(ValueError, match="1-40"):
            encode_qr("TEST", version=41)

    def test_encode_qr_invalid_ec(self):
        with pytest.raises(ValueError, match="error_correction"):
            encode_qr("TEST", error_correction='X')


class TestSymbolicPrior:
    def test_upca_prior_shape(self):
        r = upca_symbolic_prior()
        assert r.shape == (UPCA_N,)
        assert np.all((r >= 0) & (r <= 1))

    def test_upca_prior_guards_pinned(self):
        r = upca_symbolic_prior()
        # Start guard: 1,0,1 -> inverted: 0,1,0
        assert r[0] == 0.0
        assert r[1] == 1.0
        assert r[2] == 0.0

    def test_upca_prior_data_uncertain(self):
        r = upca_symbolic_prior()
        # Data modules should be 0.5
        assert r[3] == 0.5
        assert r[10] == 0.5

    def test_qr_prior_shape(self):
        for version in [1, 2, 5]:
            r = qr_symbolic_prior(version)
            size = qr_size(version)
            assert r.shape == (size, size)


# ---------------------------------------------------------------------------
# Signal processing tests
# ---------------------------------------------------------------------------

class TestSignalOps:
    def test_upscale_downscale_adjoint(self):
        x = np.array([1.0, 0.0, 1.0, 0.5])
        m = 3
        up = upscale(x, m)
        assert up.shape == (12,)
        down = downscale_sum(up, m)
        np.testing.assert_allclose(down, x * m)

    def test_upscale_2d(self):
        x = np.eye(3)
        up = upscale(x, 2)
        assert up.shape == (6, 6)

    def test_gaussian_kernel_normalized(self):
        k = gaussian_kernel_1d(21, 1.0)
        assert abs(k.sum() - 1.0) < 1e-10

    def test_gaussian_kernel_2d_normalized(self):
        k = gaussian_kernel_2d(11, 2.0)
        assert abs(k.sum() - 1.0) < 1e-10

    def test_gaussian_kernel_invalid(self):
        with pytest.raises(ValueError, match="odd"):
            gaussian_kernel_1d(10, 1.0)
        with pytest.raises(ValueError, match="positive"):
            gaussian_kernel_1d(11, 0.0)

    def test_box_kernel_normalized(self):
        k = box_kernel_1d(9)
        assert abs(k.sum() - 1.0) < 1e-10
        assert len(k) == 9

    def test_motion_kernel_normalized(self):
        k = motion_kernel(11, angle_deg=45)
        assert abs(k.sum() - 1.0) < 1e-10
        assert k.shape == (11, 11)

    def test_blur_signal_preserves_range(self):
        signal = np.random.rand(100)
        k = gaussian_kernel_1d(5, 1.0)
        b = blur_signal(signal, k, noise_var=0.01)
        assert np.all(b >= 0) and np.all(b <= 1)


# ---------------------------------------------------------------------------
# Core algorithm tests
# ---------------------------------------------------------------------------

class TestEstimation:
    def test_estimate_image_returns_valid(self):
        """estimate_image should return values in [0, 1]."""
        np.random.seed(42)
        b = np.random.rand(30)
        c = np.ones(3) / 3
        r = np.full(10, 0.5)
        x_hat = estimate_image(b, c, r, m=3, alpha=1e6, maxiter=50)
        assert x_hat.shape == (10,)
        assert np.all(x_hat >= 0) and np.all(x_hat <= 1)

    def test_estimate_image_return_dual(self):
        b = np.random.rand(30)
        c = np.ones(3) / 3
        r = np.full(10, 0.5)
        x_hat, lam = estimate_image(b, c, r, m=3, alpha=1e6,
                                     maxiter=10, return_dual=True)
        assert x_hat.shape == (10,)
        assert lam.shape == (30,)

    def test_estimate_image_warm_start_converges(self):
        """Warm-starting with previous lambda should give same result."""
        # Use a real barcode signal so the optimization is well-conditioned
        random.seed(42)
        digits = ''.join([str(random.randint(0, 9)) for _ in range(11)])
        digits += compute_check_digit(digits)
        x = encode_upca(digits)
        kernel = gaussian_kernel_1d(9, 1.0)
        b, r_inv = prepare_signal_1d(x, 3, kernel)
        x1, lam1 = estimate_image(b, kernel, r_inv, m=3, alpha=1e6,
                                    maxiter=200, return_dual=True)
        x2, lam2 = estimate_image(b, kernel, r_inv, m=3, alpha=1e6,
                                    maxiter=200, lam0=lam1, return_dual=True)
        np.testing.assert_allclose(x1, x2, atol=1e-4)

    def test_estimate_image_validates_inputs(self):
        b = np.random.rand(30)
        c = np.ones(3) / 3
        r = np.full(10, 0.5)
        with pytest.raises(ValueError, match="m must be >= 1"):
            estimate_image(b, c, r, m=0, alpha=1e6)
        with pytest.raises(ValueError, match="alpha must be positive"):
            estimate_image(b, c, r, m=3, alpha=-1)

    def test_estimate_kernel_return_dual(self):
        np.random.seed(42)
        b = np.random.rand(30)
        x_hat = np.random.rand(10)
        c_hat, lam = estimate_kernel(b, x_hat, 5, m=3, beta=1e6,
                                      maxiter=10, return_dual=True)
        assert c_hat.shape == (5,)
        assert abs(c_hat.sum() - 1.0) < 0.1  # Softmax output sums ~1


# ---------------------------------------------------------------------------
# End-to-end recovery tests (the important ones)
# ---------------------------------------------------------------------------

def _make_barcode_and_blur(seed, blur_width, kernel_fn, sigma=1.0, m=3, noise=0.0):
    """Helper: generate a random barcode, blur it, return (x, b, r_inv)."""
    random.seed(seed)
    digits = ''.join([str(random.randint(0, 9)) for _ in range(11)])
    digits += compute_check_digit(digits)
    x = encode_upca(digits)
    kernel = kernel_fn(blur_width) if 'sigma' not in kernel_fn.__code__.co_varnames else kernel_fn(blur_width, sigma)
    b, r_inv = prepare_signal_1d(x, m, kernel, noise)
    return x, b, r_inv, digits


class TestRecoveryBarcode:
    """Tests based on benchmark-validated parameter ranges."""

    @pytest.mark.parametrize("blur_width", [5, 9, 13, 17, 21])
    def test_gaussian_noiseless(self, blur_width):
        """Gaussian blur should decode at all tested widths with no noise."""
        successes = 0
        n = 3
        for seed in range(n):
            x, b, _, digits = _make_barcode_and_blur(
                seed + 100, blur_width, gaussian_kernel_1d)
            result = recover_barcode(b, m=3, max_kernel_width=max(blur_width + 4, 15))
            if result.success:
                successes += 1
        assert successes >= 2, f"Expected >= 2/{n} success at width {blur_width}, got {successes}"

    @pytest.mark.parametrize("noise", [0.005, 0.01])
    def test_gaussian_with_noise(self, noise):
        """Gaussian blur width 9 should mostly decode even with noise."""
        successes = 0
        n = 5
        for seed in range(n):
            x, b, _, digits = _make_barcode_and_blur(
                seed + 200, 9, gaussian_kernel_1d, noise=noise)
            result = recover_barcode(b, m=3)
            if result.success:
                successes += 1
        assert successes >= 3, f"Expected >= 3/{n} at noise={noise}, got {successes}"

    def test_box_blur_width_5(self):
        """Box blur width 5 should decode reliably."""
        successes = 0
        n = 5
        for seed in range(n):
            x, b, _, digits = _make_barcode_and_blur(
                seed + 300, 5, box_kernel_1d)
            result = recover_barcode(b, m=3)
            if result.success:
                successes += 1
        assert successes >= 4, f"Expected >= 4/{n}, got {successes}"

    def test_result_has_correct_fields(self):
        """DeblurResult should have all expected fields."""
        x, b, _, digits = _make_barcode_and_blur(42, 9, gaussian_kernel_1d)
        result = recover_barcode(b, m=3)
        assert isinstance(result, DeblurResult)
        assert isinstance(result.success, bool)
        assert result.modules is not None
        assert result.modules.shape == (UPCA_N,)
        assert result.iterations > 0
        assert result.kernel_width > 0

    def test_recover_barcode_validates_input(self):
        with pytest.raises(ValueError, match="1D signal"):
            recover_barcode(np.zeros((10, 10)), m=3)
        with pytest.raises(ValueError, match="empty"):
            recover_barcode(np.array([]), m=3)
        with pytest.raises(ValueError, match="m must be"):
            recover_barcode(np.zeros(100), m=0)
        with pytest.raises(ValueError, match="odd integer"):
            recover_barcode(np.zeros(100), m=3, max_kernel_width=4)


class TestRecoveryQR:
    def test_recover_qr_validates_input(self):
        with pytest.raises(ValueError, match="2D image"):
            recover_qr(np.zeros(100), m=5)
        with pytest.raises(ValueError, match="empty"):
            recover_qr(np.zeros((0, 0)), m=5)
        with pytest.raises(ValueError, match="version"):
            recover_qr(np.zeros((100, 100)), m=5, version=0)


class TestFullAlgorithm:
    def test_entropic_blind_deblur_validates_inputs(self):
        with pytest.raises(ValueError, match="1D or 2D"):
            entropic_blind_deblur(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), m=3)
        with pytest.raises(ValueError, match="same number of dimensions"):
            entropic_blind_deblur(np.zeros(10), np.zeros((3, 3)), m=3)
        with pytest.raises(ValueError, match="m must be"):
            entropic_blind_deblur(np.zeros(10), np.zeros(10), m=0)

    def test_snapshots_collected(self):
        """entropic_blind_deblur should populate snapshots list."""
        x, b, r_inv, _ = _make_barcode_and_blur(42, 9, gaussian_kernel_1d)
        extract = make_extract_fn_1d(UPCA_QUIET_ZONE, UPCA_N)
        snaps = []
        entropic_blind_deblur(
            b, r_inv, m=3, max_kernel_width=11,
            extract_fn=extract, snapshots=snaps, verbose=False,
        )
        assert len(snaps) > 0
        x_snap, c_snap, kw, it = snaps[0]
        assert x_snap.shape == (UPCA_N,)


# ---------------------------------------------------------------------------
# Image input tests
# ---------------------------------------------------------------------------

class TestImageInput:
    def test_load_image_from_array(self):
        img = np.random.randint(0, 256, (100, 200), dtype=np.uint8)
        result = load_image(img)
        assert result.dtype == np.float64
        assert result.shape == (100, 200)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_load_image_float_passthrough(self):
        img = np.random.rand(50, 100)
        result = load_image(img)
        np.testing.assert_array_almost_equal(result, img)

    def test_extract_scanline_shape(self):
        img = np.random.rand(50, 339)  # 113 * 3 = 339
        scanline = extract_barcode_scanline(img, m=3)
        assert scanline.shape == (339,)

    def test_extract_scanline_rejects_1d(self):
        with pytest.raises(ValueError, match="2D"):
            extract_barcode_scanline(np.zeros(100), m=3)


# ---------------------------------------------------------------------------
# Streaming components tests
# ---------------------------------------------------------------------------

class TestStreaming:
    def test_frame_state_is_warm(self):
        s = FrameState()
        assert not s.is_warm
        s.lam_image = np.zeros(10)
        s.kernel_width = 5
        assert s.is_warm

    @_requires_cv2
    def test_sharpness_filter_score(self):
        sf = SharpnessFilter(percentile=0.3)
        # Sharp image (edges)
        sharp = np.zeros((100, 100), dtype=np.uint8)
        sharp[40:60, 40:60] = 255
        score_sharp = sf.score(sharp)

        # Blurry image (uniform)
        blurry = np.full((100, 100), 128, dtype=np.uint8)
        score_blurry = sf.score(blurry)

        assert score_sharp > score_blurry

    @_requires_cv2
    def test_sharpness_filter_accepts_early_frames(self):
        """Should accept all frames until enough history is built."""
        sf = SharpnessFilter(percentile=0.3, history_size=30)
        img = np.full((50, 50), 128, dtype=np.uint8)
        # First few frames should always be accepted
        for _ in range(4):
            assert sf.should_process(img)

    def test_sharpness_filter_invalid_percentile(self):
        with pytest.raises(ValueError, match="percentile"):
            SharpnessFilter(percentile=0.0)
        with pytest.raises(ValueError, match="percentile"):
            SharpnessFilter(percentile=1.0)
