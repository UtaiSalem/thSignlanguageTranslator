"""ทำให้พิมพ์ภาษาไทยลงหน้าจอได้เสมอ ไม่ว่าจะถูกเรียกจากที่ไหน

ปัญหา: บน Windows ถ้าไม่ได้ตั้งตัวแปร PYTHONIOENCODING ไว้ Python จะใช้
รหัสอักขระของระบบ (cp1252 หรือ cp874) ซึ่งเขียนภาษาไทยไม่ได้ โปรแกรมจะพังด้วย
UnicodeEncodeError ตั้งแต่บรรทัด print แรก

`run.bat` ตั้งค่าให้แล้ว แต่ถ้ามีใครเรียกสคริปต์ตรง ๆ หรือเรียกจากเครื่องมืออื่น
(เช่น ตัวรันเซิร์ฟเวอร์ของ IDE) ค่านั้นจะหายไป จึงต้องกันไว้ในโค้ดด้วย
"""

from __future__ import annotations

import sys


def enable_utf8_output() -> None:
    """บังคับให้ stdout และ stderr เป็น UTF-8 เรียกซ้ำได้ไม่มีผลข้างเคียง"""
    for stream in (sys.stdout, sys.stderr):
        # ถูกเปลี่ยนทางไปที่อื่น (เช่น ไฟล์หรือ pipe) อาจไม่มีเมธอดนี้
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
