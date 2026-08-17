"""วาดข้อความภาษาไทยลงบนเฟรมของ OpenCV

ทำไมต้องมีไฟล์นี้: `cv2.putText` รองรับแค่ ASCII เท่านั้น ถ้าส่งภาษาไทยเข้าไป
จะได้เครื่องหมาย "?" เรียงกันเป็นแถว ทางแก้คือแปลงเฟรมเป็นภาพของ Pillow
วาดข้อความด้วยฟอนต์ไทยจริง ๆ แล้วแปลงกลับมาเป็นอาเรย์ของ OpenCV
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


@lru_cache(maxsize=None)
def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """หาฟอนต์ไทยตัวแรกที่มีอยู่ในเครื่อง แล้วแคชไว้ตามขนาด"""
    for candidate in config.THAI_FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)

    print(
        "คำเตือน: ไม่พบฟอนต์ภาษาไทยในเครื่อง ข้อความไทยอาจแสดงเป็นกล่องสี่เหลี่ยม\n"
        "         แก้ได้โดยเพิ่ม path ของไฟล์ฟอนต์ลงใน THAI_FONT_CANDIDATES ใน config.py"
    )
    return ImageFont.load_default(size=size)


def draw_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    font_size: int = 32,
    color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] | None = None,
    padding: int = 8,
) -> np.ndarray:
    """วาดข้อความ (ไทยหรืออังกฤษ) ลงบนเฟรม BGR แล้วคืนเฟรมใหม่

    สีที่รับเข้ามาเป็น BGR ให้เข้ากันกับโค้ด OpenCV ส่วนอื่น ๆ
    `position` คือมุมซ้ายบนของข้อความ
    """
    if not text:
        return frame

    image = Image.fromarray(frame[:, :, ::-1])   # BGR -> RGB
    draw = ImageDraw.Draw(image)
    font = _load_font(font_size)
    x, y = position

    if bg_color is not None:
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
        draw.rectangle(
            (left - padding, top - padding, right + padding, bottom + padding),
            fill=tuple(reversed(bg_color)),
        )

    draw.text((x, y), text, font=font, fill=tuple(reversed(color)))
    return np.asarray(image)[:, :, ::-1].copy()   # RGB -> BGR


def text_size(text: str, font_size: int = 32) -> tuple[int, int]:
    """คืนขนาด (กว้าง, สูง) ของข้อความเป็นพิกเซล ใช้จัดวางเลย์เอาต์"""
    font = _load_font(font_size)
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top
