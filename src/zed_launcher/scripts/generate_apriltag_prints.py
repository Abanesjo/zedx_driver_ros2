#!/usr/bin/env python3
"""Generate actual-size printable AprilTag pages as PNG and PDF files."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


MM_PER_INCH = 25.4
CM_PER_INCH = 2.54
MARKER_BORDER_BITS = 1


@dataclass(frozen=True)
class PaperSpec:
    name: str
    width_mm: float
    height_mm: float


PAPERS = {
    "A4": PaperSpec("A4", 210.0, 297.0),
    "LETTER": PaperSpec("Letter", 215.9, 279.4),
}


@dataclass(frozen=True)
class GeneratedPage:
    tag_id: int
    marker_size_cm: float
    png_path: Path
    page_size_px: tuple[int, int]
    marker_box: tuple[int, int, int, int]
    quiet_zone_px: int
    module_count: int


@dataclass(frozen=True)
class GenerationResult:
    dictionary_name: str
    pages: tuple[GeneratedPage, ...]
    pdf_path: Path


_DICTIONARIES = {
    "APRILTAG_16H5": ("APRILTAG_16h5", "DICT_APRILTAG_16h5"),
    "APRILTAG_25H9": ("APRILTAG_25h9", "DICT_APRILTAG_25h9"),
    "APRILTAG_36H10": ("APRILTAG_36h10", "DICT_APRILTAG_36h10"),
    "APRILTAG_36H11": ("APRILTAG_36h11", "DICT_APRILTAG_36h11"),
}


def _normalize_dictionary_name(name: str) -> str:
    normalized = name.strip().upper()
    if normalized.startswith("DICT_"):
        normalized = normalized[5:]
    if normalized in {"16H5", "25H9", "36H10", "36H11"}:
        normalized = f"APRILTAG_{normalized}"
    if normalized not in _DICTIONARIES:
        supported = ", ".join(value[0] for value in _DICTIONARIES.values())
        raise ValueError(
            f"Unsupported dictionary '{name}'. Supported dictionaries: "
            f"{supported}"
        )
    return normalized


def get_dictionary(name: str):
    """Return ``(canonical_name, OpenCV dictionary)`` for an AprilTag family."""
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "OpenCV was built without the aruco module; install opencv-contrib-python"
        )

    normalized = _normalize_dictionary_name(name)
    canonical, constant_name = _DICTIONARIES[normalized]
    if not hasattr(cv2.aruco, constant_name):
        raise RuntimeError(
            f"This OpenCV build does not provide {constant_name}"
        )
    dictionary_id = getattr(cv2.aruco, constant_name)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    elif hasattr(cv2.aruco, "Dictionary_get"):
        dictionary = cv2.aruco.Dictionary_get(dictionary_id)
    else:
        raise RuntimeError("This OpenCV aruco API cannot load dictionaries")
    return canonical, dictionary


def generate_marker_image(dictionary, tag_id: int, side_pixels: int) -> np.ndarray:
    """Generate a marker using either the modern or legacy OpenCV API."""
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(
            dictionary, int(tag_id), int(side_pixels)
        )
        if marker is not None:
            return np.asarray(marker, dtype=np.uint8)

        marker = np.empty((side_pixels, side_pixels), dtype=np.uint8)
        cv2.aruco.generateImageMarker(
            dictionary, int(tag_id), int(side_pixels), marker,
            MARKER_BORDER_BITS,
        )
        return marker

    if hasattr(cv2.aruco, "drawMarker"):
        try:
            marker = cv2.aruco.drawMarker(
                dictionary, int(tag_id), int(side_pixels)
            )
            if marker is not None:
                return np.asarray(marker, dtype=np.uint8)
        except TypeError:
            pass

        marker = np.empty((side_pixels, side_pixels), dtype=np.uint8)
        cv2.aruco.drawMarker(
            dictionary, int(tag_id), int(side_pixels), marker,
            MARKER_BORDER_BITS,
        )
        return marker

    raise RuntimeError(
        "This OpenCV aruco API provides neither generateImageMarker nor drawMarker"
    )


def _pixels_from_mm(length_mm: float, dpi: int) -> int:
    return int(round(length_mm * dpi / MM_PER_INCH))


def _pixels_from_cm(length_cm: float, dpi: int) -> int:
    return int(round(length_cm * dpi / CM_PER_INCH))


def _font(points: float, dpi: int, bold: bool = False):
    size = max(8, int(round(points * dpi / 72.0)))
    names = (
        ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if bold
        else ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    page_width: int,
    y: int,
    text: str,
    font,
    fill: int = 0,
) -> None:
    if hasattr(draw, "textbbox"):
        box = draw.textbbox((0, 0), text, font=font)
        width = box[2] - box[0]
    else:
        width = draw.textsize(text, font=font)[0]
    draw.text(((page_width - width) // 2, y), text, fill=fill, font=font)


def _draw_cut_guides(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    dpi: int,
) -> None:
    left, top, right, bottom = box
    length = _pixels_from_mm(6.0, dpi)
    gap = _pixels_from_mm(1.5, dpi)
    line_width = max(1, _pixels_from_mm(0.25, dpi))

    for x, direction in ((left, -1), (right, 1)):
        start_x = x + direction * gap
        end_x = x + direction * (gap + length)
        for y in (top, bottom):
            draw.line((start_x, y, end_x, y), fill=0, width=line_width)

    for y, direction in ((top, -1), (bottom, 1)):
        start_y = y + direction * gap
        end_y = y + direction * (gap + length)
        for x in (left, right):
            draw.line((x, start_y, x, end_y), fill=0, width=line_width)


def _draw_scale(draw: ImageDraw.ImageDraw, paper: PaperSpec, dpi: int) -> None:
    page_width = _pixels_from_mm(paper.width_mm, dpi)
    page_height = _pixels_from_mm(paper.height_mm, dpi)
    scale_length = _pixels_from_cm(10.0, dpi)
    center_x = page_width // 2
    left = center_x - scale_length // 2
    right = left + scale_length
    y = page_height - _pixels_from_mm(18.0, dpi)
    tick = _pixels_from_mm(4.0, dpi)
    line_width = max(1, _pixels_from_mm(0.35, dpi))
    draw.line((left, y, right, y), fill=0, width=line_width)
    draw.line((left, y - tick, left, y + tick), fill=0, width=line_width)
    draw.line((right, y - tick, right, y + tick), fill=0, width=line_width)
    _draw_centered_text(
        draw,
        page_width,
        y - _pixels_from_mm(9.0, dpi),
        "10 cm verification scale",
        _font(9.0, dpi),
    )


def _render_page(
    dictionary_name: str,
    dictionary,
    tag_id: int,
    marker_size_cm: float,
    paper: PaperSpec,
    dpi: int,
):
    page_width = _pixels_from_mm(paper.width_mm, dpi)
    page_height = _pixels_from_mm(paper.height_mm, dpi)
    marker_size_px = _pixels_from_cm(marker_size_cm, dpi)
    module_count = int(dictionary.markerSize) + 2 * MARKER_BORDER_BITS
    quiet_zone_px = max(1, int(round(marker_size_px / module_count)))
    print_square_px = marker_size_px + 2 * quiet_zone_px

    safe_margin_px = _pixels_from_mm(8.0, dpi)
    content_top = _pixels_from_mm(35.0, dpi)
    content_bottom = page_height - _pixels_from_mm(42.0, dpi)
    available_width = page_width - 2 * safe_margin_px
    available_height = content_bottom - content_top
    if print_square_px > available_width or print_square_px > available_height:
        max_square_mm = min(
            available_width * MM_PER_INCH / dpi,
            available_height * MM_PER_INCH / dpi,
        )
        raise ValueError(
            f"A {marker_size_cm:g} cm marker plus its one-module quiet zone "
            f"does not fit on {paper.name}; available square is "
            f"{max_square_mm / 10.0:.2f} cm"
        )
    if marker_size_px < module_count * 4:
        raise ValueError(
            "The selected size and DPI provide fewer than four pixels per module"
        )

    print_left = (page_width - print_square_px) // 2
    print_top = content_top + (available_height - print_square_px) // 2
    marker_left = print_left + quiet_zone_px
    marker_top = print_top + quiet_zone_px
    marker_box = (
        marker_left,
        marker_top,
        marker_left + marker_size_px,
        marker_top + marker_size_px,
    )
    print_box = (
        print_left,
        print_top,
        print_left + print_square_px,
        print_top + print_square_px,
    )

    page = Image.new("L", (page_width, page_height), color=255)
    marker = generate_marker_image(dictionary, tag_id, marker_size_px)
    page.paste(Image.fromarray(marker, mode="L"), (marker_left, marker_top))

    draw = ImageDraw.Draw(page)
    _draw_centered_text(
        draw,
        page_width,
        _pixels_from_mm(9.0, dpi),
        f"{dictionary_name}  |  ID {tag_id}",
        _font(16.0, dpi, bold=True),
    )
    module_size_cm = marker_size_cm / module_count
    _draw_centered_text(
        draw,
        page_width,
        _pixels_from_mm(18.0, dpi),
        (
            f"Outer black marker: {marker_size_cm:.2f} cm  |  "
            f"quiet zone: 1 module ({module_size_cm:.2f} cm)"
        ),
        _font(9.5, dpi),
    )
    _draw_centered_text(
        draw,
        page_width,
        _pixels_from_mm(26.0, dpi),
        "ACTUAL SIZE: print at 100%; disable Fit, Shrink, or Scale to page.",
        _font(10.0, dpi, bold=True),
    )
    _draw_cut_guides(draw, print_box, dpi)
    _draw_scale(draw, paper, dpi)

    return page, GeneratedPage(
        tag_id=tag_id,
        marker_size_cm=marker_size_cm,
        png_path=Path(),
        page_size_px=(page_width, page_height),
        marker_box=marker_box,
        quiet_zone_px=quiet_zone_px,
        module_count=module_count,
    )


def _expand_sizes(sizes_cm: Sequence[float], count: int) -> list[float]:
    sizes = [float(size) for size in sizes_cm]
    if len(sizes) == 1:
        sizes *= count
    elif len(sizes) != count:
        raise ValueError(
            "--sizes-cm must contain one shared size or one size per tag ID"
        )
    if any(not math.isfinite(size) or size <= 0.0 for size in sizes):
        raise ValueError("All marker sizes must be positive finite values")
    return sizes


def _size_token(size_cm: float) -> str:
    token = f"{size_cm:.3f}".rstrip("0").rstrip(".")
    return token.replace(".", "p")


def generate_prints(
    dictionary_name: str = "APRILTAG_36h11",
    ids: Sequence[int] = (0, 1),
    sizes_cm: Sequence[float] = (12.0,),
    paper_name: str = "A4",
    dpi: int = 600,
    output_dir: Path | str = Path("apriltag_prints"),
) -> GenerationResult:
    """Generate one printable PNG per ID and a combined multi-page PDF."""
    tag_ids = [int(tag_id) for tag_id in ids]
    if not tag_ids:
        raise ValueError("At least one tag ID is required")
    if len(set(tag_ids)) != len(tag_ids):
        raise ValueError("Tag IDs must be unique")
    if not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("DPI must be a positive integer")

    canonical_name, dictionary = get_dictionary(dictionary_name)
    dictionary_size = int(dictionary.bytesList.shape[0])
    for tag_id in tag_ids:
        if tag_id < 0 or tag_id >= dictionary_size:
            raise ValueError(
                f"Tag ID {tag_id} is outside {canonical_name}'s range "
                f"0..{dictionary_size - 1}"
            )

    paper_key = paper_name.strip().upper()
    if paper_key not in PAPERS:
        raise ValueError("Paper must be A4 or Letter")
    paper = PAPERS[paper_key]
    marker_sizes = _expand_sizes(sizes_cm, len(tag_ids))

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", canonical_name.lower()).strip("_")
    rendered_pages = []
    generated_pages = []
    try:
        for tag_id, marker_size_cm in zip(tag_ids, marker_sizes):
            page_image, metadata = _render_page(
                canonical_name,
                dictionary,
                tag_id,
                marker_size_cm,
                paper,
                dpi,
            )
            filename = (
                f"{slug}_id_{tag_id}_{_size_token(marker_size_cm)}cm_"
                f"{paper.name.lower()}_{dpi}dpi.png"
            )
            png_path = destination / filename
            page_image.save(png_path, format="PNG", dpi=(dpi, dpi))
            rendered_pages.append(page_image)
            generated_pages.append(
                GeneratedPage(
                    tag_id=metadata.tag_id,
                    marker_size_cm=metadata.marker_size_cm,
                    png_path=png_path,
                    page_size_px=metadata.page_size_px,
                    marker_box=metadata.marker_box,
                    quiet_zone_px=metadata.quiet_zone_px,
                    module_count=metadata.module_count,
                )
            )

        ids_token = "-".join(str(tag_id) for tag_id in tag_ids)
        pdf_path = destination / (
            f"{slug}_ids_{ids_token}_{paper.name.lower()}_{dpi}dpi.pdf"
        )
        dither_none = (
            Image.Dither.NONE if hasattr(Image, "Dither") else Image.NONE
        )
        pdf_pages = [
            page.convert("1", dither=dither_none) for page in rendered_pages
        ]
        try:
            pdf_pages[0].save(
                pdf_path,
                format="PDF",
                save_all=True,
                append_images=pdf_pages[1:],
                resolution=float(dpi),
                title=f"{canonical_name} printable tags",
                subject="Print at actual size (100%)",
            )
        finally:
            for page in pdf_pages:
                page.close()
    finally:
        for page in rendered_pages:
            page.close()

    return GenerationResult(
        dictionary_name=canonical_name,
        pages=tuple(generated_pages),
        pdf_path=pdf_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate actual-size AprilTag page PNGs and a combined PDF. "
            "The requested size is the outer black marker, excluding the "
            "one-module white quiet zone."
        )
    )
    parser.add_argument(
        "--dictionary",
        default="APRILTAG_36h11",
        help="AprilTag family (default: APRILTAG_36h11)",
    )
    parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        default=[0, 1],
        metavar="ID",
        help="tag IDs, separated by spaces (default: 0 1)",
    )
    parser.add_argument(
        "--sizes-cm",
        "--size-cm",
        type=float,
        nargs="+",
        default=[12.0],
        metavar="CM",
        help=(
            "outer black-marker sizes: one shared value or one per ID "
            "(default: 12)"
        ),
    )
    parser.add_argument(
        "--paper",
        type=lambda value: value.strip().upper(),
        choices=("A4", "LETTER"),
        default="A4",
        help="paper size: A4 or Letter (default: A4)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="raster and PDF print resolution (default: 600)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("apriltag_prints"),
        help="destination directory (default: ./apriltag_prints)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = generate_prints(
            dictionary_name=args.dictionary,
            ids=args.ids,
            sizes_cm=args.sizes_cm,
            paper_name=args.paper,
            dpi=args.dpi,
            output_dir=args.output_dir,
        )
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    for page in result.pages:
        print(page.png_path)
    print(result.pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
