"""อ่าน/เขียนไฟล์ชุดข้อมูล landmark (CSV)

เก็บเป็น CSV ธรรมดาโดยตั้งใจ เพื่อให้เปิดดูด้วย Excel หรือ Notepad ได้
นักเรียนจะได้เห็นว่า "ข้อมูลที่โมเดลเรียน" หน้าตาเป็นอย่างไรจริง ๆ
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from .features import FEATURE_DIMS, feature_column_names


def append_sample(csv_path: Path, label: str, vector: np.ndarray) -> None:
    """เพิ่มข้อมูล 1 แถวลงท้ายไฟล์ทันที (ไม่รอปิดโปรแกรม ข้อมูลจึงไม่หายถ้าโปรแกรมดับ)"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["label"] + feature_column_names())
        writer.writerow([label] + [f"{value:.6f}" for value in vector])


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """โหลดชุดข้อมูลทั้งหมด คืน (X, y) — X คือ features, y คือชื่อคำ"""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"ยังไม่มีไฟล์ข้อมูล {csv_path}\n"
            f"ให้รัน `python -m src.collect` เพื่อเก็บข้อมูลจากกล้องก่อน"
        )

    labels: list[str] = []
    rows: list[list[float]] = []

    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            labels.append(row["label"])
            rows.append([float(row[name]) for name in feature_column_names()])

    if not rows:
        raise ValueError(f"ไฟล์ {csv_path} ยังไม่มีข้อมูล")

    X = np.array(rows, dtype=np.float32)
    if X.shape[1] != FEATURE_DIMS:
        raise ValueError(f"จำนวน feature ไม่ตรง: พบ {X.shape[1]} ควรเป็น {FEATURE_DIMS}")

    return X, np.array(labels)


def count_samples(csv_path: Path) -> Counter:
    """นับจำนวนตัวอย่างของแต่ละคำ ใช้แสดงความคืบหน้าตอนเก็บข้อมูล"""
    counts: Counter = Counter()
    if not csv_path.exists():
        return counts

    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            counts[row["label"]] += 1
    return counts


def drop_last_sample(csv_path: Path, label: str) -> bool:
    """ลบตัวอย่างล่าสุดของคำที่ระบุ ใช้ตอนเผลอเก็บท่าผิด คืน True ถ้าลบสำเร็จ"""
    if not csv_path.exists():
        return False

    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        lines = fh.readlines()

    for index in range(len(lines) - 1, 0, -1):   # ข้ามบรรทัดหัวตาราง
        if lines[index].startswith(f"{label},"):
            del lines[index]
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                fh.writelines(lines)
            return True

    return False
