"""Generate synthetic test videos of blurry barcodes for pipeline validation.

Creates short mp4 clips with known ground truth so we can verify the
streaming pipeline end-to-end without a physical camera.

Usage:
    python tests/generate_test_video.py              # default: 3 seconds, Gaussian blur
    python tests/generate_test_video.py --frames 90  # 3 seconds at 30fps
    python tests/generate_test_video.py --kernel box --blur-width 7
"""

import argparse
import random
import sys
import os

import numpy as np

# Ensure zbar is findable on macOS
if os.path.exists('/opt/homebrew/lib/libzbar.dylib'):
    os.environ.setdefault('DYLD_LIBRARY_PATH', '/opt/homebrew/lib')

try:
    import cv2
except ImportError:
    print("opencv-python required: pip install opencv-python")
    sys.exit(1)

from deblurrinator.entropic_deblur import (
    encode_upca, compute_check_digit, upscale,
    gaussian_kernel_1d, box_kernel_1d,
    blur_signal, UPCA_QUIET_ZONE, UPCA_N,
)


def generate_barcode_frame(x, m, height, pad_h, pad_w):
    """Render a 1D barcode as a grayscale image with padding."""
    row = upscale(x, m)
    bar_img = np.tile(row, (height, 1))
    # Pad to make a reasonable frame size
    frame = np.ones((height + 2 * pad_h, len(row) + 2 * pad_w))
    frame[pad_h:pad_h + height, pad_w:pad_w + len(row)] = bar_img
    return frame


def make_test_video(output_path, n_frames=90, fps=30, blur_width=15,
                    kernel_type='gaussian', sigma=1.0, noise_var=0.001,
                    m=3, drift_pixels=2):
    """Generate a synthetic test video of a blurry barcode.

    The barcode drifts slightly between frames to simulate hand tremor,
    and blur/noise vary slightly to simulate focus changes.
    """
    # Generate barcode
    random.seed(42)
    digits = ''.join([str(random.randint(0, 9)) for _ in range(11)])
    digits += compute_check_digit(digits)
    x = encode_upca(digits)
    print(f"Ground truth barcode: {digits}")

    # Build the blur kernel
    if kernel_type == 'gaussian':
        kernel = gaussian_kernel_1d(blur_width, sigma)
    elif kernel_type == 'box':
        kernel = box_kernel_1d(blur_width)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    # Render clean barcode
    n_modules = UPCA_N + 2 * UPCA_QUIET_ZONE
    bar_height = 80
    pad_h, pad_w = 40, 30
    frame_h = bar_height + 2 * pad_h
    frame_w = n_modules * m + 2 * pad_w

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h), False)
    if not out.isOpened():
        raise RuntimeError(f"Could not create video writer for {output_path}")

    print(f"Generating {n_frames} frames at {fps}fps ({n_frames/fps:.1f}s)")
    print(f"Frame size: {frame_w}x{frame_h}, kernel: {kernel_type} w={blur_width}")

    np.random.seed(42)
    for i in range(n_frames):
        # Slight horizontal drift to simulate hand tremor
        shift = int(drift_pixels * np.sin(2 * np.pi * i / n_frames * 3))

        # Vary noise slightly across frames
        frame_noise = noise_var * (0.5 + np.random.rand())

        # Build padded barcode with quiet zones
        x_inv = 1.0 - x
        x_padded = np.concatenate([
            np.zeros(UPCA_QUIET_ZONE), x_inv, np.zeros(UPCA_QUIET_ZONE)
        ])
        signal = upscale(x_padded, m)

        # Apply blur and noise
        blurred = blur_signal(signal, kernel, noise_var=frame_noise)

        # Render as image
        bar_img = np.tile(blurred, (bar_height, 1))
        frame = np.ones((frame_h, frame_w))

        # Apply shift
        x_start = pad_w + shift
        x_end = x_start + len(blurred)
        if x_start >= 0 and x_end <= frame_w:
            frame[pad_h:pad_h + bar_height, x_start:x_end] = bar_img

        # Convert to uint8
        frame_u8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        out.write(frame_u8)

    out.release()
    print(f"Saved {output_path}")
    return digits


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic barcode test video')
    parser.add_argument('--output', type=str, default='tests/test_barcode.mp4')
    parser.add_argument('--frames', type=int, default=90)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--blur-width', type=int, default=15)
    parser.add_argument('--kernel', type=str, default='gaussian',
                        choices=['gaussian', 'box'])
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--noise', type=float, default=0.001)
    parser.add_argument('--m', type=int, default=3)
    args = parser.parse_args()

    digits = make_test_video(
        args.output, n_frames=args.frames, fps=args.fps,
        blur_width=args.blur_width, kernel_type=args.kernel,
        sigma=args.sigma, noise_var=args.noise, m=args.m,
    )

    # Verify the video is readable
    cap = cv2.VideoCapture(args.output)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Verified: {n} frames readable")
    print(f"\nTo test the pipeline:")
    print(f"  python -c \"from deblurrinator.streaming import process_video; "
          f"process_video('{args.output}', barcode_type='1d', m={args.m})\"")


if __name__ == '__main__':
    main()
