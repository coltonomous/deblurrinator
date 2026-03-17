"""Real image input pipeline for barcode deblurring.

Bridges actual photographs of blurry barcodes/QR codes to the deblurring
algorithm. Handles image loading, ROI detection, scanline extraction, and
signal normalization.

    from deblurrinator import deblur_from_image
    result = deblur_from_image("blurry_barcode.jpg", barcode_type='1d', m=3)
    if result.success:
        print(result.data)

Note: Automatic ROI detection is best-effort and works poorly on heavily
blurred images. For reliable results, pass a pre-cropped image containing
only the barcode region.
"""

from pathlib import Path
from typing import Union, Optional

import numpy as np

from .entropic_deblur import UPCA_N, UPCA_QUIET_ZONE, QR_QUIET_ZONE, qr_size
from .deblur_recovery import recover_barcode, recover_qr, DeblurResult

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    _HAS_PYZBAR = True
except ImportError:
    _HAS_PYZBAR = False


def _require_cv2():
    if not _HAS_CV2:
        raise ImportError(
            "opencv-python is required for image input. "
            "Install it with: pip install opencv-python"
        )


def load_image(source: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Load an image as a grayscale float64 array in [0, 1].

    Parameters
    ----------
    source : str, Path, or ndarray
        File path to an image, or a numpy array (uint8 or float).

    Returns
    -------
    ndarray
        2D grayscale image, float64 in [0, 1].
    """
    if isinstance(source, np.ndarray):
        img = source.copy()
    else:
        _require_cv2()
        path = str(source)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")

    # Convert to 2D grayscale if needed
    if img.ndim == 3:
        _require_cv2()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize to [0, 1] float64
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    return np.clip(img.astype(np.float64), 0, 1)


def _pyzbar_detect(image: np.ndarray, barcode_type: str) -> list:
    """Try to detect barcodes using pyzbar. Returns candidates list or empty."""
    if not _HAS_PYZBAR:
        return []

    img_u8 = (image * 255).astype(np.uint8)
    results = pyzbar_decode(img_u8)
    if not results:
        return []

    candidates = []
    for r in results:
        x, y, w, h = r.rect
        if w < 20 or h < 20:
            continue
        aspect = w / h if h > 0 else 0

        if barcode_type == '1d' and aspect < 2.0:
            continue
        if barcode_type == '2d' and (aspect < 0.5 or aspect > 2.0):
            continue

        candidates.append({
            'bbox': (x, y, w, h),
            'aspect_ratio': aspect,
        })

    candidates.sort(key=lambda c: c['bbox'][2] * c['bbox'][3], reverse=True)
    return candidates


def _gradient_detect(image: np.ndarray, barcode_type: str) -> list:
    """Fallback ROI detection using gradient analysis and morphology."""
    _require_cv2()

    img_u8 = (image * 255).astype(np.uint8)

    # Gradient magnitude
    grad_x = cv2.Sobel(img_u8, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_u8, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    gradient = (gradient / gradient.max() * 255).astype(np.uint8) if gradient.max() > 0 else gradient.astype(np.uint8)

    # Threshold and morphological close to merge barcode bars
    _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Dilate to merge nearby components
    closed = cv2.dilate(closed, None, iterations=4)
    closed = cv2.erode(closed, None, iterations=4)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 20 or h < 20:
            continue
        aspect = w / h if h > 0 else 0

        if barcode_type == '1d' and aspect < 2.0:
            continue
        if barcode_type == '2d' and (aspect < 0.5 or aspect > 2.0):
            continue

        candidates.append({
            'bbox': (x, y, w, h),
            'aspect_ratio': aspect,
        })

    candidates.sort(key=lambda c: c['bbox'][2] * c['bbox'][3], reverse=True)
    return candidates


def detect_barcode_roi(image: np.ndarray, barcode_type: str = 'any'):
    """Detect barcode regions in an image.

    First attempts detection via pyzbar (fast, accurate when the barcode
    is readable). Falls back to gradient-based morphological detection
    for blurred images that pyzbar can't decode.

    Parameters
    ----------
    image : ndarray
        Grayscale image, float64 in [0, 1].
    barcode_type : str
        'any', '1d', or '2d'. Filters candidates by aspect ratio.

    Returns
    -------
    list of dict
        Each dict has 'bbox' (x, y, w, h) and 'aspect_ratio'.
        Sorted by area (largest first).
    """
    candidates = _pyzbar_detect(image, barcode_type)
    if candidates:
        return candidates
    return _gradient_detect(image, barcode_type)


def extract_barcode_scanline(
    image: np.ndarray,
    m: int = 3,
    expected_modules: int = UPCA_N + 2 * UPCA_QUIET_ZONE,
) -> np.ndarray:
    """Extract a 1D barcode scanline from a grayscale image.

    Averages all rows to produce a single scanline (reduces noise),
    then resizes to match the expected signal length for the deblurring
    algorithm.

    Parameters
    ----------
    image : ndarray
        2D grayscale image of the barcode region, float64 in [0, 1].
        Should be cropped to contain only the barcode.
    m : int
        Pixels per barcode module.
    expected_modules : int
        Expected number of modules including quiet zones.
        Default: UPCA_N + 2 * UPCA_QUIET_ZONE = 113.

    Returns
    -------
    ndarray
        1D signal of length expected_modules * m, ready for recover_barcode().
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    # Average rows to get a 1D scanline
    scanline = image.mean(axis=0)

    # Resize to expected length
    target_len = expected_modules * m
    if len(scanline) != target_len:
        _require_cv2()
        scanline = cv2.resize(
            scanline.reshape(1, -1), (target_len, 1),
            interpolation=cv2.INTER_LINEAR,
        ).ravel()

    # Ensure [0, 1] range
    scanline = np.clip(scanline, 0, 1)
    return scanline.astype(np.float64)


def extract_qr_region(
    image: np.ndarray,
    m: int = 5,
    version: Optional[int] = None,
) -> tuple:
    """Extract a QR code region from a grayscale image.

    Crops to square (center), resizes to the expected signal dimensions.

    Parameters
    ----------
    image : ndarray
        2D grayscale image of the QR region, float64 in [0, 1].
    m : int
        Pixels per QR module.
    version : int or None
        QR version (1-40). If None, estimated from image size and m.

    Returns
    -------
    (ndarray, int)
        Resized image ready for recover_qr(), and the estimated version.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    # Crop to square from center
    h, w = image.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    square = image[y0:y0 + side, x0:x0 + side]

    # Estimate version from image size if not given
    if version is None:
        estimated_modules = side / m - 2 * QR_QUIET_ZONE
        version = max(1, round((estimated_modules - 17) / 4))

    size = qr_size(version)
    target = (size + 2 * QR_QUIET_ZONE) * m

    if square.shape[0] != target:
        _require_cv2()
        square = cv2.resize(square, (target, target), interpolation=cv2.INTER_LINEAR)

    return np.clip(square, 0, 1).astype(np.float64), version


def deblur_from_image(
    source: Union[str, Path, np.ndarray],
    barcode_type: str = '1d',
    m: int = 3,
    version: Optional[int] = None,
    auto_detect: bool = False,
    **kwargs,
) -> DeblurResult:
    """Deblur a barcode from a photograph.

    This is the main entry point for real-image deblurring. It loads the
    image, optionally detects the barcode ROI, extracts the signal, and
    runs the recovery algorithm.

    Parameters
    ----------
    source : str, Path, or ndarray
        Image file path or numpy array.
    barcode_type : str
        '1d' for UPC-A barcodes, '2d' for QR codes.
    m : int
        Pixels per barcode/QR module. For real images, this depends on
        the resolution and distance. You may need to experiment.
    version : int or None
        QR version (for barcode_type='2d'). If None, estimated from size.
    auto_detect : bool
        If True, attempt to detect and crop the barcode region
        automatically. If False (default), assume the input is already
        cropped to the barcode area.
    **kwargs
        Passed to recover_barcode() or recover_qr() (e.g., alpha, beta,
        max_kernel_width).

    Returns
    -------
    DeblurResult
        .success is True if decoding succeeded, .data has the decoded string.
    """
    image = load_image(source)

    if auto_detect:
        rois = detect_barcode_roi(image, barcode_type=barcode_type)
        if not rois:
            return DeblurResult(success=False)
        x, y, w, h = rois[0]['bbox']
        image = image[y:y+h, x:x+w]

    if barcode_type == '1d':
        signal = extract_barcode_scanline(image, m=m)
        return recover_barcode(signal, m=m, **kwargs)
    elif barcode_type == '2d':
        region, est_version = extract_qr_region(image, m=m, version=version)
        return recover_qr(region, m=m, version=est_version, **kwargs)
    else:
        raise ValueError(f"barcode_type must be '1d' or '2d', got {barcode_type!r}")
