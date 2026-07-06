"""
Utility helpers: Cyrillic text rendering on OpenCV frames, geometry helpers.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_font_cache: dict = {}


def _get_font(font_size: int):
    if font_size in _font_cache:
        return _font_cache[font_size]
    for path in (
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            font = ImageFont.truetype(path, font_size)
            _font_cache[font_size] = font
            return font
        except (IOError, OSError):
            pass
    font = ImageFont.load_default()
    _font_cache[font_size] = font
    return font


def put_cyrillic_text(
    frame: np.ndarray,
    text: str,
    pos: tuple,
    color_bgr: tuple,
    font_size: int = 36,
    bg_color: tuple = (0, 0, 0),
) -> np.ndarray:
    """
    Draw Cyrillic (or any Unicode) text on an OpenCV BGR frame using PIL.
    Only a small strip around the text region is converted to PIL and back,
    keeping the operation fast.
    """
    font = _get_font(font_size)
    dummy = Image.new("RGB", (1, 1))
    bbox = ImageDraw.Draw(dummy).textbbox(pos, text, font=font)
    pad = 8
    strip_y1 = max(0, bbox[1] - pad)
    strip_y2 = min(frame.shape[0], bbox[3] + pad + 1)
    strip_x1 = max(0, bbox[0] - pad)
    strip_x2 = min(frame.shape[1], bbox[2] + pad + 1)

    strip = frame[strip_y1:strip_y2, strip_x1:strip_x2]
    pil_strip = Image.fromarray(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_strip)

    adj_pos = (pos[0] - strip_x1, pos[1] - strip_y1)
    adj_bbox = (
        bbox[0] - strip_x1, bbox[1] - strip_y1,
        bbox[2] - strip_x1, bbox[3] - strip_y1,
    )
    draw.rectangle(
        [adj_bbox[0] - pad, adj_bbox[1] - pad, adj_bbox[2] + pad, adj_bbox[3] + pad],
        fill=(bg_color[2], bg_color[1], bg_color[0]),
    )
    draw.text(adj_pos, text, font=font,
              fill=(color_bgr[2], color_bgr[1], color_bgr[0]))

    frame[strip_y1:strip_y2, strip_x1:strip_x2] = cv2.cvtColor(
        np.array(pil_strip), cv2.COLOR_RGB2BGR)
    return frame


def is_point_near_box(px: float, py: float, box: list, margin: int = 50) -> bool:
    """Returns True if point (px, py) is within margin pixels of box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box
    return (x1 - margin <= px <= x2 + margin) and (y1 - margin <= py <= y2 + margin)


def box_gap(box_a: list, box_b: list) -> float:
    """
    Smallest gap (in pixels) between two axis-aligned boxes [x1, y1, x2, y2].

    Returns 0.0 if the boxes overlap, otherwise the Euclidean distance between
    their nearest edges/corners.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return (dx * dx + dy * dy) ** 0.5


def is_box_near_box(box_a: list, box_b: list, margin: float = 0.0) -> bool:
    """
    True if box_a is within `margin` pixels of box_b (overlap counts as near).

    Used to express spatial rules like "chock is placed at the wheel".
    """
    return box_gap(box_a, box_b) <= margin
