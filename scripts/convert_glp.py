# -*- coding: utf-8 -*-
"""
Convert .glp layout files to PNG images for NeuralILT pipeline.

GLP files from LithoBench StdMetal/StdContact are ASCII text files
containing polygon coordinates (PGON lines). This script parses the
polygons and rasterizes them onto a grid to produce grayscale PNGs.

Format example:
    BEGIN     /* The metadata are invalid */
    EQUIV  1  1000  MICRON  +X,+Y
    CNAME Temp_Top
    LEVEL M1
    CELL Temp_Top PRIME
       PGON N M1  60 505 60 880 760 880 760 610 ...
    ENDMSG

Usage:
    python scripts/convert_glp.py data/raw/StdMetal
    python scripts/convert_glp.py data/raw/StdMetal data/raw/StdContact
    python scripts/convert_glp.py --probe-only data/raw/StdMetal
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_glp_file(filepath):
    """Parse a .glp file and extract polygon coordinates.

    Returns a list of polygons, where each polygon is a list of (x, y) tuples.
    """
    polygons = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('PGON'):
                continue

            # PGON N M1  x1 y1 x2 y2 x3 y3 ...
            parts = line.split()
            # skip 'PGON', 'N', 'M1' (or similar layer info)
            # find where the numbers start
            coords = []
            for i, p in enumerate(parts):
                if i < 3:  # skip PGON, N, M1
                    continue
                try:
                    coords.append(int(p))
                except ValueError:
                    try:
                        coords.append(float(p))
                    except ValueError:
                        continue

            # coords are x1, y1, x2, y2, ...
            if len(coords) >= 4 and len(coords) % 2 == 0:
                poly = [(coords[i], coords[i+1])
                        for i in range(0, len(coords), 2)]
                polygons.append(poly)

    return polygons


def rasterize_polygons(polygons, image_size=2048, margin=10):
    """Render polygons onto a raster grid.

    Args:
        polygons: list of polygons (each is list of (x,y) tuples)
        image_size: output image size (square)
        margin: pixel margin around the design

    Returns:
        numpy array (image_size x image_size), values 0 or 255
    """
    if not polygons:
        return np.zeros((image_size, image_size), dtype=np.uint8)

    # find bounding box of all polygons
    all_x = [x for poly in polygons for x, y in poly]
    all_y = [y for poly in polygons for x, y in poly]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # compute scale to fit in image_size with margin
    width = max_x - min_x
    height = max_y - min_y
    if width == 0 or height == 0:
        return np.zeros((image_size, image_size), dtype=np.uint8)

    usable = image_size - 2 * margin
    scale = min(usable / width, usable / height)

    # create image and draw polygons
    img = Image.new('L', (image_size, image_size), 0)
    draw = ImageDraw.Draw(img)

    for poly in polygons:
        # transform coordinates to pixel space
        pixel_poly = []
        for x, y in poly:
            px = margin + (x - min_x) * scale
            py = margin + (y - min_y) * scale
            pixel_poly.append((px, py))

        if len(pixel_poly) >= 3:
            draw.polygon(pixel_poly, fill=255)

    return np.array(img)


def convert_glp_to_png(glp_path, png_path, image_size=2048):
    """Convert a single .glp file to PNG."""
    polygons = parse_glp_file(glp_path)
    if not polygons:
        # empty layout — save blank image
        img = np.zeros((image_size, image_size), dtype=np.uint8)
    else:
        img = rasterize_polygons(polygons, image_size=image_size)

    Image.fromarray(img).save(png_path)
    return len(polygons)


def convert_directory(data_dir, image_size=2048):
    """Convert all .glp files in a directory to PNG.

    Handles both flat directories (StdMetal/*.glp) and
    subdirectory structures (StdMetal/target/*.glp).
    """
    data_dir = Path(data_dir)
    name = data_dir.name

    print(f"\n{'=' * 50}")
    print(f"Converting {name} (.glp -> .png)")
    print(f"{'=' * 50}")

    converted = 0
    failed = 0

    # check for subdirectory structure (target/ litho/)
    subdirs = ['target', 'litho']
    has_subdirs = any((data_dir / s).exists() for s in subdirs)

    if has_subdirs:
        dirs_to_process = [(data_dir / s) for s in subdirs
                           if (data_dir / s).exists()]
    else:
        # flat directory — all .glp files are in the root
        # create target/ subdir and convert there
        dirs_to_process = [data_dir]

    for subdir in dirs_to_process:
        glp_files = sorted([f for f in subdir.iterdir()
                           if f.suffix.lower() == '.glp'])

        if not glp_files:
            print(f"  [SKIP] {subdir.name}/: no .glp files")
            continue

        print(f"\n  {subdir.relative_to(data_dir.parent)}: {len(glp_files)} .glp files")

        # probe first file
        polys = parse_glp_file(glp_files[0])
        print(f"  Sample: {glp_files[0].name} -> {len(polys)} polygons")

        for i, glp_path in enumerate(glp_files):
            png_path = glp_path.with_suffix('.png')
            try:
                n_polys = convert_glp_to_png(glp_path, png_path,
                                              image_size=image_size)
                converted += 1
            except Exception as e:
                print(f"  FAIL: {glp_path.name}: {e}")
                failed += 1

            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(glp_files)} converted...")

        print(f"  Done: {converted} converted in {subdir.name}/")

    # if flat directory, create target/ structure for our pipeline
    if not has_subdirs and converted > 0:
        target_dir = data_dir / "target"
        target_dir.mkdir(exist_ok=True)
        for f in data_dir.glob("*.png"):
            f.rename(target_dir / f.name)
        for f in data_dir.glob("*.glp"):
            f.rename(target_dir / f.name)
        # create empty litho/ (we only have layouts, no masks for StdMetal)
        (data_dir / "litho").mkdir(exist_ok=True)
        print(f"\n  Moved files to {name}/target/ (created target/litho structure)")

    print(f"\n  Total: {converted} converted, {failed} failed")
    return converted, failed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert .glp layout files to PNG images")
    parser.add_argument('dirs', nargs='+',
                        help='Directories containing .glp files')
    parser.add_argument('--probe-only', action='store_true',
                        help='Only parse the first file, do not convert')
    parser.add_argument('--size', type=int, default=2048,
                        help='Output image size (default: 2048)')
    args = parser.parse_args()

    total_converted = 0
    total_failed = 0

    for d in args.dirs:
        if args.probe_only:
            data_dir = Path(d)
            glp_files = list(data_dir.rglob('*.glp'))
            if glp_files:
                print(f"\nProbing: {glp_files[0]}")
                polys = parse_glp_file(glp_files[0])
                print(f"  Polygons: {len(polys)}")
                for i, poly in enumerate(polys[:3]):
                    print(f"  Polygon {i}: {len(poly)} vertices, "
                          f"bbox ({min(x for x,y in poly)},{min(y for x,y in poly)}) "
                          f"to ({max(x for x,y in poly)},{max(y for x,y in poly)})")
            else:
                print(f"No .glp files found in {d}")
        else:
            c, f = convert_directory(d, image_size=args.size)
            total_converted += c
            total_failed += f

    if not args.probe_only:
        print(f"\n{'=' * 50}")
        print(f"All done: {total_converted} converted, {total_failed} failed")
        print(f"{'=' * 50}")
