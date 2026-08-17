"""อ่าน/เขียนไฟล์ชุดข้อมูล landmark (CSV)

เก็บเป็น CSV ธรรมดาโดยตั้งใจ เพื่อให้เปิดดูด้วย Excel หรือ Notepad ได้
นักเรียนจะได้เห็นว่า "ข้อมูลที่โมเดลเรียน" หน้าตาเป็นอย่างไรจริง ๆ
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from .features import FEATURE_DIMS, LEGACY_FEATURE_DIMS, feature_column_names


def dataset_is_legacy(csv_path: Path) -> bool:
    """เดาจากหัวตารางว่าไฟล์นี้เป็นชุดข้อมูลรุ่นก่อนมีท่อนคู่มือหรือไม่

    ดูจากจำนวนคอลัมน์ ไม่ได้ดูจากเลขรุ่นเพราะไฟล์ CSV ไม่ได้บันทึกเลขรุ่นไว้
    """
    if not csv_path.exists():
        return False
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh), [])
    return len(header) == 1 + LEGACY_FEATURE_DIMS and "Pair_present" not in header


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

    if dataset_is_legacy(csv_path):
        raise ValueError(
            f"ไฟล์ {csv_path} เป็นชุดข้อมูลรุ่นเก่า ({LEGACY_FEATURE_DIMS} มิติ "
            f"ตอนนี้ใช้ {FEATURE_DIMS} มิติ)\n"
            f"แปลงให้ทันสมัยก่อนด้วย `run transfer convert-legacy` "
            f"(คำสั่งนี้สำรองไฟล์เดิมให้อัตโนมัติ)"
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
        rows = list(csv.reader(fh))

    # ต้องแกะแถวด้วย csv.reader แล้วเทียบช่องแรก ไม่ใช่เทียบว่าบรรทัดขึ้นต้นด้วย
    # "ชื่อคำ," เพราะคำที่มีลูกน้ำหรือเครื่องหมายคำพูดจะถูก CSV ครอบ quote ไว้
    # (เช่น "ก,ข") การเทียบสตริงตรง ๆ จะหาไม่เจอแล้วบอกว่าไม่มีตัวอย่างให้ลบ
    for index in range(len(rows) - 1, 0, -1):   # ข้ามแถวหัวตาราง
        if rows[index] and rows[index][0] == label:
            del rows[index]
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows(rows)
            return True

    return False
