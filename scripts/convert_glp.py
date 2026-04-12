# -*- coding: utf-8 -*-
"""
Convert .glp files from LithoBench StdMetal/StdContact to PNG images.

GLP files from LithoBench are raw binary grid patterns. This script
auto-detects the format and converts them to grayscale PNG images.

Usage:
    python scripts/convert_glp.py data/raw/StdMetal
    python scripts/convert_glp.py data/raw/StdContact
    python scripts/convert_glp.py data/raw/StdMetal data/raw/StdContact

The script will:
1. Probe the first .glp file to detect dimensions
2. Convert all .glp files to .png in the same directory
3. Report success/failure counts
"""

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def probe_glp_file(filepath):
    """Try to detect the format of a .glp file.

    Returns (width, height, dtype, header_size) or None if unrecognized.
    """
    filepath = Path(filepath)
    file_size = filepath.stat().st_size

    with open(filepath, 'rb') as f:
        header = f.read(64)

    print(f"  File: {filepath.name}")
    print(f"  Size: {file_size} bytes")
    print(f"  First 32 bytes (hex): {header[:32].hex()}")
    print(f"  First 16 bytes (ascii): {repr(header[:16])}")

    # Strategy 1: Check if it's a raw float32 grid (no header)
    # Common sizes: 2048x2048, 1024x1024, 512x512, 256x256
    for dim in [2048, 1024, 512, 256]:
        expected_f32 = dim * dim * 4  # float32
        expected_f64 = dim * dim * 8  # float64
        expected_u8 = dim * dim       # uint8

        if file_size == expected_f32:
            print(f"  Detected: {dim}x{dim} float32 (no header)")
            return dim, dim, np.float32, 0
        elif file_size == expected_f64:
            print(f"  Detected: {dim}x{dim} float64 (no header)")
            return dim, dim, np.float64, 0
        elif file_size == expected_u8:
            print(f"  Detected: {dim}x{dim} uint8 (no header)")
            return dim, dim, np.uint8, 0

    # Strategy 2: Check if first 8 bytes are dimensions (int32 width, int32 height)
    if len(header) >= 8:
        w, h = struct.unpack('<ii', header[:8])
        if 64 <= w <= 4096 and 64 <= h <= 4096:
            remaining = file_size - 8
            pixels = w * h
            if remaining == pixels * 4:
                print(f"  Detected: {w}x{h} float32 (8-byte header)")
                return w, h, np.float32, 8
            elif remaining == pixels * 8:
                print(f"  Detected: {w}x{h} float64 (8-byte header)")
                return w, h, np.float64, 8
            elif remaining == pixels:
                print(f"  Detected: {w}x{h} uint8 (8-byte header)")
                return w, h, np.uint8, 8

    # Strategy 3: Try as text file (space/newline separated values)
    try:
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            values = first_line.split()
            if len(values) > 10:
                # count lines to get height
                f.seek(0)
                lines = f.readlines()
                h = len(lines)
                w = len(lines[0].strip().split())
                print(f"  Detected: {w}x{h} text grid")
                return w, h, 'text', 0
    except (UnicodeDecodeError, ValueError):
        pass

    # Strategy 4: Try to guess from file size (assume square, float32)
    pixels = file_size // 4
    dim = int(np.sqrt(pixels))
    if dim * dim * 4 == file_size and dim >= 64:
        print(f"  Guessing: {dim}x{dim} float32 (from file size)")
        return dim, dim, np.float32, 0

    print(f"  ERROR: Could not detect format")
    return None


def read_glp_file(filepath, fmt):
    """Read a .glp file and return a numpy array."""
    w, h, dtype, header_size = fmt

    if dtype == 'text':
        data = np.loadtxt(filepath)
        return data.astype(np.float32)

    with open(filepath, 'rb') as f:
        if header_size > 0:
            f.read(header_size)
        raw = f.read()

    if dtype == np.float32:
        arr = np.frombuffer(raw, dtype=np.float32)
    elif dtype == np.float64:
        arr = np.frombuffer(raw, dtype=np.float64).astype(np.float32)
    elif dtype == np.uint8:
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 255.0
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    return arr.reshape(h, w)


def convert_glp_to_png(glp_path, png_path, fmt):
    """Convert a single .glp file to PNG."""
    arr = read_glp_file(glp_path, fmt)

    # normalize to [0, 255]
    if arr.max() > 1.0:
        arr = arr / arr.max()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)

    Image.fromarray(arr, mode='L').save(png_path)


def convert_directory(data_dir):
    """Convert all .glp files in target/ and litho/ subdirs to PNG."""
    data_dir = Path(data_dir)
    name = data_dir.name

    print(f"\n{'=' * 50}")
    print(f"Converting {name}")
    print(f"{'=' * 50}")

    converted = 0
    failed = 0

    for subdir_name in ['target', 'litho']:
        subdir = data_dir / subdir_name
        if not subdir.exists():
            print(f"  [SKIP] {subdir_name}/ not found")
            continue

        glp_files = sorted([f for f in subdir.iterdir()
                           if f.suffix.lower() == '.glp'])

        if not glp_files:
            print(f"  [SKIP] No .glp files in {subdir_name}/")
            continue

        print(f"\n  {subdir_name}/: {len(glp_files)} .glp files")

        # probe first file to detect format
        fmt = probe_glp_file(glp_files[0])
        if fmt is None:
            print(f"  ERROR: Cannot detect .glp format. Skipping {subdir_name}/")
            failed += len(glp_files)
            continue

        print(f"  Converting {len(glp_files)} files...")

        for i, glp_path in enumerate(glp_files):
            png_path = glp_path.with_suffix('.png')
            try:
                convert_glp_to_png(glp_path, png_path, fmt)
                converted += 1
            except Exception as e:
                print(f"  FAIL: {glp_path.name}: {e}")
                failed += 1

            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(glp_files)} converted...")

        print(f"  Done: {converted} converted in {subdir_name}/")

    print(f"\n  Total: {converted} converted, {failed} failed")
    return converted, failed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert .glp files to PNG images")
    parser.add_argument('dirs', nargs='+',
                        help='Directories containing target/ and litho/ with .glp files')
    parser.add_argument('--probe-only', action='store_true',
                        help='Only probe the first file, do not convert')
    args = parser.parse_args()

    total_converted = 0
    total_failed = 0

    for d in args.dirs:
        if args.probe_only:
            data_dir = Path(d)
            for subdir in ['target', 'litho']:
                glp_files = list((data_dir / subdir).glob('*.glp'))
                if glp_files:
                    probe_glp_file(glp_files[0])
        else:
            c, f = convert_directory(d)
            total_converted += c
            total_failed += f

    if not args.probe_only:
        print(f"\n{'=' * 50}")
        print(f"All done: {total_converted} converted, {total_failed} failed")
        print(f"{'=' * 50}")
