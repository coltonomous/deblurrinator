"""Streaming and video-mode barcode deblurring.

Processes video frames with warm-started optimization for near-real-time
barcode recovery from blurry camera feeds.

    from deblurrinator.streaming import live_camera, process_video

    # Live camera deblurring
    live_camera(source=0, barcode_type='1d', m=3)

    # Batch process a video file
    results = process_video("blurry_scan.mp4", barcode_type='1d', m=3)

Key ideas:
- Warm-starting: reuse the previous frame's dual variables (lambda) to
  accelerate L-BFGS convergence from ~200 iterations to ~20-50.
- Frame selection: skip blurry frames using Laplacian variance scoring.
- Asynchronous processing: capture thread + worker thread so the display
  never blocks on optimization.
"""

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from .entropic_deblur import (
    UPCA_N,
    UPCA_QUIET_ZONE,
    QR_QUIET_ZONE,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    upca_symbolic_prior,
    qr_symbolic_prior,
    qr_size,
    estimate_image,
    estimate_kernel,
    make_check_fn,
    make_qr_check_fn,
    make_extract_fn_1d,
    make_extract_fn_2d,
)
from .deblur_recovery import (
    DeblurResult,
    RECOVERY_GTOL,
    RECOVERY_MAXITER,
    RECOVERY_INNER_ITERS,
)
from .image_input import load_image, extract_barcode_scanline, extract_qr_region

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def _require_cv2():
    if not _HAS_CV2:
        raise ImportError(
            "opencv-python is required for streaming mode. "
            "Install it with: pip install opencv-python"
        )


# --- Warm-start tuning ---
# When warm-starting from a previous frame's solution, L-BFGS converges
# much faster. These reduced limits avoid wasting time when the solution
# is already close.
WARM_MAXITER = 50
WARM_GTOL = 1e-6

# After this many consecutive warm-start failures, reset to cold start.
MAX_WARM_FAILURES = 3


@dataclass
class FrameState:
    """Cached state from the previous frame for warm-starting."""
    x_hat: Optional[np.ndarray] = None
    c_hat: Optional[np.ndarray] = None
    lam_image: Optional[np.ndarray] = None
    lam_kernel: Optional[np.ndarray] = None
    kernel_width: int = 0
    consecutive_failures: int = 0

    @property
    def is_warm(self):
        return self.lam_image is not None and self.kernel_width > 0


@dataclass
class StreamResult:
    """Result from processing a single video frame."""
    frame_id: int
    result: DeblurResult
    sharpness_score: float
    processing_time_ms: float
    warm_started: bool


class SharpnessFilter:
    """Frame selection via Laplacian variance.

    Tracks a rolling window of sharpness scores and only passes frames
    that exceed an adaptive threshold (percentile of recent history).
    """

    def __init__(self, percentile: float = 0.3, history_size: int = 30):
        """
        Parameters
        ----------
        percentile : float
            Fraction of frames to accept (0.3 = top 70% sharpest).
        history_size : int
            Number of recent scores to track for adaptive thresholding.
        """
        if not 0 < percentile < 1:
            raise ValueError(f"percentile must be in (0, 1), got {percentile}")
        self._percentile = percentile
        self._history = deque(maxlen=history_size)

    def score(self, frame: np.ndarray) -> float:
        """Laplacian variance of the frame. Higher = sharper."""
        _require_cv2()
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(frame, cv2.CV_64F).var()

    def should_process(self, frame: np.ndarray) -> bool:
        """True if frame is sharp enough relative to recent history."""
        s = self.score(frame)
        self._history.append(s)

        # Always process if we don't have enough history yet
        if len(self._history) < 5:
            return True

        threshold = np.percentile(list(self._history), self._percentile * 100)
        return s >= threshold

    def score_and_filter(self, frame: np.ndarray) -> tuple:
        """Return (score, should_process) in one call."""
        s = self.score(frame)
        self._history.append(s)
        if len(self._history) < 5:
            return s, True
        threshold = np.percentile(list(self._history), self._percentile * 100)
        return s, s >= threshold


def _warm_recovery_loop(b, r, m, state, alpha, beta, max_kernel_width,
                        check_fn, extract_fn):
    """Alternating estimation with warm-start from previous frame.

    If state is warm, starts at the previously successful kernel width
    with cached dual variables and reduced iteration limits. Falls back
    to a cold start if warm attempts fail.

    Returns (DeblurResult, updated FrameState).
    """
    is_2d = b.ndim == 2
    total_iters = 0

    if state.is_warm and state.consecutive_failures < MAX_WARM_FAILURES:
        # --- Warm path: start at the known-good kernel width ---
        kw = state.kernel_width
        kernel_shape = (kw, kw) if is_2d else kw
        x_hat = state.x_hat.copy()

        # Try a single pass with warm-started L-BFGS
        c_hat, lam_k = estimate_kernel(
            b, x_hat, kernel_shape, m, beta,
            gtol=WARM_GTOL, maxiter=WARM_MAXITER,
            lam0=state.lam_kernel, return_dual=True,
        )
        x_hat, lam_i = estimate_image(
            b, c_hat, r, m, alpha,
            gtol=WARM_GTOL, maxiter=WARM_MAXITER,
            lam0=state.lam_image, return_dual=True,
        )
        total_iters += 1

        x_thresh = (x_hat > 0.5).astype(np.float64)
        x_modules = extract_fn(x_thresh)

        if check_fn is not None:
            decoded = check_fn(x_modules)
            if decoded:
                new_state = FrameState(
                    x_hat=x_hat, c_hat=c_hat,
                    lam_image=lam_i, lam_kernel=lam_k,
                    kernel_width=kw, consecutive_failures=0,
                )
                result = DeblurResult(
                    success=True, data=decoded,
                    modules=x_modules, kernel=c_hat,
                    iterations=total_iters, kernel_width=kw,
                )
                return result, new_state

        # Warm attempt failed — try one more pass at adjacent kernel widths
        for kw_try in [kw - 2, kw + 2, kw]:
            if kw_try < 3 or kw_try > max_kernel_width:
                continue
            ks = (kw_try, kw_try) if is_2d else kw_try
            c_hat = estimate_kernel(b, x_hat, ks, m, beta,
                                    gtol=WARM_GTOL, maxiter=WARM_MAXITER)
            x_hat = estimate_image(b, c_hat, r, m, alpha,
                                   gtol=WARM_GTOL, maxiter=WARM_MAXITER)
            total_iters += 1

            x_thresh = (x_hat > 0.5).astype(np.float64)
            x_modules = extract_fn(x_thresh)
            if check_fn is not None:
                decoded = check_fn(x_modules)
                if decoded:
                    new_state = FrameState(
                        x_hat=x_hat, c_hat=c_hat,
                        kernel_width=kw_try, consecutive_failures=0,
                    )
                    result = DeblurResult(
                        success=True, data=decoded,
                        modules=x_modules, kernel=c_hat,
                        iterations=total_iters, kernel_width=kw_try,
                    )
                    return result, new_state

        # All warm attempts failed
        state.consecutive_failures += 1

    # --- Cold path: full coarse-to-fine sweep ---
    x_hat = r.copy()
    c_hat = None
    lam_i, lam_k = None, None

    for i in range(1, max_kernel_width // 2 + 1):
        kw = 2 * i + 1
        if kw > min(b.shape):
            break
        kernel_shape = (kw, kw) if is_2d else kw

        for j in range(RECOVERY_INNER_ITERS):
            c_hat, lam_k = estimate_kernel(
                b, x_hat, kernel_shape, m, beta,
                gtol=RECOVERY_GTOL, maxiter=RECOVERY_MAXITER,
                return_dual=True,
            )
            x_hat, lam_i = estimate_image(
                b, c_hat, r, m, alpha,
                gtol=RECOVERY_GTOL, maxiter=RECOVERY_MAXITER,
                return_dual=True,
            )
            total_iters += 1

            x_thresh = (x_hat > 0.5).astype(np.float64)
            x_modules = extract_fn(x_thresh)

            if check_fn is not None:
                decoded = check_fn(x_modules)
                if decoded:
                    new_state = FrameState(
                        x_hat=x_hat, c_hat=c_hat,
                        lam_image=lam_i, lam_kernel=lam_k,
                        kernel_width=kw, consecutive_failures=0,
                    )
                    result = DeblurResult(
                        success=True, data=decoded,
                        modules=x_modules, kernel=c_hat,
                        iterations=total_iters, kernel_width=kw,
                    )
                    return result, new_state

    # Failed entirely
    x_modules = extract_fn((x_hat > 0.5).astype(np.float64))
    new_state = FrameState(
        x_hat=x_hat, c_hat=c_hat,
        lam_image=lam_i, lam_kernel=lam_k,
        kernel_width=kw if c_hat is not None else 0,
        consecutive_failures=state.consecutive_failures + 1,
    )
    result = DeblurResult(
        success=False, modules=x_modules, kernel=c_hat,
        iterations=total_iters,
        kernel_width=kw if c_hat is not None else 0,
    )
    return result, new_state


def _build_pipeline_fns(barcode_type, m, version=None):
    """Build the r_padded, check_fn, and extract_fn for a barcode type."""
    if barcode_type == '1d':
        r_inv = 1.0 - upca_symbolic_prior()
        r_padded = np.concatenate([
            np.zeros(UPCA_QUIET_ZONE), r_inv, np.zeros(UPCA_QUIET_ZONE)
        ])
        check_fn = make_check_fn(m)
        extract_fn = make_extract_fn_1d(UPCA_QUIET_ZONE, UPCA_N)
        return r_padded, check_fn, extract_fn
    elif barcode_type == '2d':
        if version is None:
            raise ValueError("QR version is required for streaming mode")
        size = qr_size(version)
        r_inv = 1.0 - qr_symbolic_prior(version)
        r_padded = np.pad(r_inv, QR_QUIET_ZONE, mode='constant', constant_values=0)
        check_fn = make_qr_check_fn(m)
        extract_fn = make_extract_fn_2d(QR_QUIET_ZONE, size)
        return r_padded, check_fn, extract_fn
    else:
        raise ValueError(f"barcode_type must be '1d' or '2d', got {barcode_type!r}")


class StreamProcessor:
    """Asynchronous frame processing pipeline for barcode deblurring.

    Capture thread feeds frames to a bounded queue. Worker thread pulls
    frames and runs warm-started recovery. Display thread polls results.
    """

    def __init__(self, barcode_type='1d', m=3, version=None,
                 max_kernel_width=15, sharpness_percentile=0.3,
                 queue_size=2, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA,
                 roi=None):
        """
        Parameters
        ----------
        roi : tuple (x, y, w, h) or None
            Fixed region of interest to crop from each frame before
            processing. If None, uses the full frame.
        """
        self._barcode_type = barcode_type
        self._m = m
        self._roi = roi
        self._max_kernel_width = max_kernel_width
        self._alpha = alpha
        self._beta = beta
        self._state = FrameState()
        self._sharpness = SharpnessFilter(
            percentile=sharpness_percentile, history_size=30,
        )

        self._r_padded, self._check_fn, self._extract_fn = \
            _build_pipeline_fns(barcode_type, m, version)

        self._frame_queue = queue.Queue(maxsize=queue_size)
        self._result_queue = queue.Queue(maxsize=8)
        self._stop_event = threading.Event()
        self._worker_thread = None

    def submit_frame(self, frame: np.ndarray, frame_id: int = 0) -> bool:
        """Submit a frame for processing. Returns False if dropped."""
        score, should = self._sharpness.score_and_filter(frame)
        if not should:
            return False
        try:
            self._frame_queue.put_nowait((frame, frame_id, score))
            return True
        except queue.Full:
            # Drop the oldest frame and put the new one
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait((frame, frame_id, score))
                return True
            except queue.Full:
                return False

    def get_result(self, timeout: float = 0.01) -> Optional[StreamResult]:
        """Non-blocking fetch of the latest decode result."""
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def start(self):
        """Start the worker thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop(self):
        """Signal the worker to stop and wait for it to finish."""
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
            self._worker_thread = None

    def _preprocess_frame(self, frame):
        """Convert a camera frame to the signal expected by the algorithm."""
        img = load_image(frame)
        if self._roi is not None:
            rx, ry, rw, rh = self._roi
            img = img[ry:ry+rh, rx:rx+rw]
        if self._barcode_type == '1d':
            return extract_barcode_scanline(img, m=self._m)
        else:
            region, _ = extract_qr_region(img, m=self._m)
            return region

    def _worker_loop(self):
        """Worker thread: pull frames, run warm-started recovery."""
        while not self._stop_event.is_set():
            try:
                frame, frame_id, score = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            was_warm = self._state.is_warm

            try:
                b = self._preprocess_frame(frame)
                result, self._state = _warm_recovery_loop(
                    b, self._r_padded, self._m, self._state,
                    self._alpha, self._beta, self._max_kernel_width,
                    self._check_fn, self._extract_fn,
                )
            except Exception as e:
                result = DeblurResult(success=False)
                self._state = FrameState()  # Reset on error

            elapsed_ms = (time.perf_counter() - t0) * 1000

            stream_result = StreamResult(
                frame_id=frame_id,
                result=result,
                sharpness_score=score,
                processing_time_ms=elapsed_ms,
                warm_started=was_warm,
            )

            try:
                self._result_queue.put_nowait(stream_result)
            except queue.Full:
                # Drop oldest result
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    pass
                self._result_queue.put_nowait(stream_result)


def process_video(video_path, barcode_type='1d', m=3, version=None,
                  max_kernel_width=15, skip_frames=0,
                  sharpness_percentile=0.3,
                  alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA,
                  roi=None, verbose=True):
    """Process a video file frame-by-frame with warm-starting.

    Parameters
    ----------
    video_path : str
        Path to video file.
    barcode_type : str
        '1d' for UPC-A barcodes, '2d' for QR codes.
    m : int
        Pixels per module.
    version : int or None
        QR version (required for barcode_type='2d').
    skip_frames : int
        Only process every Nth frame (0 = process all passing sharpness).
    sharpness_percentile : float
        Fraction of frames to skip as too blurry.
    roi : tuple (x, y, w, h) or None
        Crop region of interest from each frame before processing.
    verbose : bool
        Print progress and results.

    Returns
    -------
    list of StreamResult
    """
    _require_cv2()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if verbose:
        print(f"Processing {video_path}: {total_frames} frames at {fps:.1f} fps")

    r_padded, check_fn, extract_fn = _build_pipeline_fns(barcode_type, m, version)
    sharpness = SharpnessFilter(percentile=sharpness_percentile)
    state = FrameState()
    results = []
    frame_id = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # Skip frames if requested
        if skip_frames > 0 and frame_id % (skip_frames + 1) != 0:
            continue

        # Sharpness filter
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score, should = sharpness.score_and_filter(gray)
        if not should:
            continue

        # Preprocess
        img = gray.astype(np.float64) / 255.0
        if roi is not None:
            rx, ry, rw, rh = roi
            img = img[ry:ry+rh, rx:rx+rw]
        if barcode_type == '1d':
            b = extract_barcode_scanline(img, m=m)
        else:
            b, _ = extract_qr_region(img, m=m, version=version)

        # Run warm-started recovery
        t0 = time.perf_counter()
        was_warm = state.is_warm
        result, state = _warm_recovery_loop(
            b, r_padded, m, state, alpha, beta, max_kernel_width,
            check_fn, extract_fn,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        processed += 1

        sr = StreamResult(
            frame_id=frame_id, result=result,
            sharpness_score=score, processing_time_ms=elapsed_ms,
            warm_started=was_warm,
        )
        results.append(sr)

        if verbose:
            status = result.data if result.success else "---"
            warm_tag = "W" if was_warm else "C"
            print(f"\r  frame {frame_id}/{total_frames} [{warm_tag}] "
                  f"{elapsed_ms:6.0f}ms  {status}", end='', flush=True)

    cap.release()

    if verbose:
        successes = sum(1 for r in results if r.result.success)
        warm_count = sum(1 for r in results if r.warm_started)
        avg_ms = np.mean([r.processing_time_ms for r in results]) if results else 0
        print(f"\n\nProcessed {processed}/{frame_id} frames")
        print(f"  Decoded: {successes}/{processed}")
        print(f"  Warm-started: {warm_count}/{processed}")
        print(f"  Avg processing time: {avg_ms:.0f}ms")

    return results


def live_camera(source=0, barcode_type='1d', m=3, version=None,
                max_kernel_width=15, roi=None, show_debug=False):
    """Run live camera barcode deblurring with OpenCV overlay.

    Press 'q' to quit, 'r' to reset warm-start state, 'd' to toggle debug.

    Parameters
    ----------
    source : int or str
        Camera index (0 = default) or video file path.
    barcode_type : str
        '1d' for UPC-A, '2d' for QR.
    m : int
        Pixels per module.
    version : int or None
        QR version (required for barcode_type='2d').
    roi : tuple (x, y, w, h) or None
        Fixed region of interest. If None, uses full frame.
    show_debug : bool
        Show sharpness, timing, and warm-start info in overlay.
    """
    _require_cv2()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera/video: {source}")

    processor = StreamProcessor(
        barcode_type=barcode_type, m=m, version=version,
        max_kernel_width=max_kernel_width,
    )
    processor.start()

    last_decode = None
    last_decode_time = 0
    frame_id = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            display = frame.copy()

            # Extract ROI if specified
            if roi is not None:
                rx, ry, rw, rh = roi
                roi_frame = frame[ry:ry+rh, rx:rx+rw]
                cv2.rectangle(display, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
            else:
                roi_frame = frame

            # Submit for processing
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            processor.submit_frame(gray, frame_id)

            # Check for results
            result = processor.get_result(timeout=0.001)
            if result is not None and result.result.success:
                last_decode = result
                last_decode_time = time.time()

            # Draw overlay
            h, w = display.shape[:2]

            if last_decode is not None:
                # Show decode result
                elapsed = time.time() - last_decode_time
                alpha_fade = max(0.0, 1.0 - elapsed / 5.0)  # Fade after 5s

                text = f"Decoded: {last_decode.result.data}"
                color = (0, int(255 * alpha_fade), 0)
                cv2.putText(display, text, (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.putText(display, "Scanning...", (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            if show_debug and result is not None:
                debug_lines = [
                    f"Frame: {result.frame_id}",
                    f"Sharp: {result.sharpness_score:.0f}",
                    f"Time: {result.processing_time_ms:.0f}ms",
                    f"Warm: {'Y' if result.warm_started else 'N'}",
                ]
                for i, line in enumerate(debug_lines):
                    cv2.putText(display, line, (10, 25 + i * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Deblurrinator', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                processor._state = FrameState()
                last_decode = None
                print("Reset warm-start state")
            elif key == ord('d'):
                show_debug = not show_debug

    finally:
        processor.stop()
        cap.release()
        cv2.destroyAllWindows()
