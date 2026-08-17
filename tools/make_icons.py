"""สร้างไอคอนของแอปมือถือ

รัน:  run icons

สร้างไฟล์ PNG สามขนาดลงในโฟลเดอร์ app/icons/ ตามที่ manifest.webmanifest ต้องการ
ไอคอนแบบ maskable ต้องเผื่อขอบว่างไว้ราว 20% เพราะ Android จะครอบมันด้วยรูปทรง
ของธีมเครื่อง (วงกลม สี่เหลี่ยมมนบ้าง) ถ้าวาดชิดขอบ เนื้อหาจะโดนตัด
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.console import enable_utf8_output  # noqa: E402

enable_utf8_output()

ICONS_DIR = config.ROOT / "app" / "icons"
BACKGROUND = (15, 23, 42)
ACCENT = (56, 189, 248)
TEXT = (241, 245, 249)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in config.THAI_FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def draw_icon(size: int, safe_ratio: float = 1.0) -> Image.Image:
    """วาดไอคอนหนึ่งรูป

    safe_ratio < 1 หมายถึงบีบเนื้อหาให้เล็กลงเพื่อเผื่อขอบสำหรับไอคอน maskable
    """
    image = Image.new("RGB", (size, size), BACKGROUND)
    draw = ImageDraw.Draw(image)
    center = size / 2
    content = size * safe_ratio

    # วงกลมพื้นหลังสีเน้น
    radius = content * 0.42
    draw.ellipse(
        (center - radius, center - radius, center + radius, center + radius),
        fill=(30, 41, 59),
        outline=ACCENT,
        width=max(int(content * 0.022), 2),
    )

    # รูปมืออย่างง่าย: ฝ่ามือกับสี่นิ้วชูขึ้น และนิ้วโป้งกางออก
    palm_width = content * 0.30
    palm_height = content * 0.24
    palm_top = center + content * 0.02
    draw.rounded_rectangle(
        (center - palm_width / 2, palm_top, center + palm_width / 2, palm_top + palm_height),
        radius=content * 0.06,
        fill=ACCENT,
    )

    finger_width = content * 0.055
    finger_gap = content * 0.017
    total_width = 4 * finger_width + 3 * finger_gap
    start_x = center - total_width / 2
    heights = [0.20, 0.26, 0.24, 0.18]   # ความยาวนิ้วต่างกันเล็กน้อยให้ดูเป็นมือ

    for index, height_ratio in enumerate(heights):
        x = start_x + index * (finger_width + finger_gap)
        finger_height = content * height_ratio
        draw.rounded_rectangle(
            (x, palm_top - finger_height, x + finger_width, palm_top + content * 0.02),
            radius=finger_width / 2,
            fill=ACCENT,
        )

    # นิ้วโป้งกางออกทางซ้าย
    thumb_width = content * 0.13
    thumb_height = content * 0.055
    draw.rounded_rectangle(
        (
            center - palm_width / 2 - thumb_width * 0.72,
            palm_top + palm_height * 0.30,
            center - palm_width / 2 + thumb_width * 0.28,
            palm_top + palm_height * 0.30 + thumb_height,
        ),
        radius=thumb_height / 2,
        fill=ACCENT,
    )

    # ตัวอักษรไทยกำกับด้านล่าง
    font = load_font(max(int(content * 0.115), 8))
    label = "ภาษามือ"
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    draw.text(
        (center - (right - left) / 2 - left, center + content * 0.255 - top),
        label,
        font=font,
        fill=TEXT,
    )

    return image


def main() -> int:
    ICONS_DIR.mkdir(parents=True, exist_ok=True)

    targets = [
        ("icon-192.png", 192, 1.0),
        ("icon-512.png", 512, 1.0),
        # maskable ต้องเผื่อขอบ เพราะ Android จะครอบด้วยรูปทรงของธีมเครื่อง
        ("icon-maskable-512.png", 512, 0.76),
    ]

    for filename, size, safe_ratio in targets:
        path = ICONS_DIR / filename
        draw_icon(size, safe_ratio).save(path, "PNG", optimize=True)
        print(f"  สร้าง {filename}  ({size}x{size}, {path.stat().st_size / 1024:.0f} KB)")

    print(f"\nบันทึกไอคอนลงใน {ICONS_DIR} เรียบร้อย")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
