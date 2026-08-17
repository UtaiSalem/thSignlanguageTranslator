"""แปลงจุด landmark ของมือให้เป็น "เวกเตอร์คุณลักษณะ" (feature vector)

นี่คือหัวใจของโปรเจค และเป็นเหตุผลที่วิธีนี้ใช้ข้อมูลน้อยกว่าการเทรนจากรูปภาพมาก

ปัญหาคือ พิกัดดิบจาก MediaPipe ขึ้นกับสิ่งที่ไม่เกี่ยวกับความหมายของท่ามือเลย:
    - มืออยู่ตรงไหนของเฟรม  (ตำแหน่ง)
    - มืออยู่ใกล้หรือไกลกล้อง (ขนาด)
เราจึงต้อง normalize ทิ้งสองอย่างนี้ออกไป เหลือไว้แค่ "รูปทรงของมือ" ล้วน ๆ:

    1. ย้ายจุดข้อมือ (landmark 0) ไปไว้ที่จุดกำเนิด  -> ตัดผลของตำแหน่ง
    2. หารด้วยระยะที่ไกลที่สุดจากข้อมือ              -> ตัดผลของขนาด

ผลลัพธ์: ทำท่าเดียวกันที่มุมไหนของจอ ใกล้หรือไกลแค่ไหน ก็ได้ตัวเลขชุดเดิม
"""

from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3          # x, y, z
PER_HAND_DIMS = NUM_LANDMARKS * COORDS_PER_LANDMARK + 1   # +1 = ธงบอกว่ามีมือข้างนี้ไหม
HAND_SLOTS = ("Left", "Right")   # ล็อกตำแหน่งในเวกเตอร์ให้คงที่เสมอ
FEATURE_DIMS = PER_HAND_DIMS * len(HAND_SLOTS)            # = 128


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """normalize จุด (21, 3) ของมือหนึ่งข้าง แล้วคืนเวกเตอร์แบน 63 ค่า"""
    points = np.asarray(landmarks, dtype=np.float32).reshape(NUM_LANDMARKS, COORDS_PER_LANDMARK)

    # 1) ย้ายข้อมือไปที่จุดกำเนิด
    centered = points - points[0]

    # 2) ปรับสเกลด้วยระยะที่ไกลสุดจากข้อมือ
    scale = np.linalg.norm(centered, axis=1).max()
    if scale < 1e-6:          # กันหารด้วยศูนย์ ถ้าจุดซ้อนกันหมด
        scale = 1.0

    return (centered / scale).flatten()


def build_feature_vector(hands) -> np.ndarray:
    """รวมมือที่ตรวจเจอ (0-2 ข้าง) เป็นเวกเตอร์ความยาวคงที่ 128 ค่า

    ต้องมีความยาวคงที่เพราะโมเดล machine learning รับ input ขนาดตายตัว
    ถ้าเจอมือข้างเดียว ช่องของอีกข้างจะเป็นศูนย์ทั้งหมด และธงบอกการมีอยู่ = 0
    """
    vector = np.zeros(FEATURE_DIMS, dtype=np.float32)

    for hand in hands:
        if hand.handedness not in HAND_SLOTS:
            continue
        slot = HAND_SLOTS.index(hand.handedness)
        start = slot * PER_HAND_DIMS
        vector[start:start + PER_HAND_DIMS - 1] = normalize_landmarks(hand.landmarks)
        vector[start + PER_HAND_DIMS - 1] = 1.0   # ธง: มือข้างนี้ปรากฏอยู่

    return vector


def mirror_feature_vector(vector: np.ndarray) -> np.ndarray:
    """สร้างเวกเตอร์ของภาพสะท้อนกระจก — ใช้เพิ่มข้อมูลตอนเทรน (data augmentation)

    ท่ามือส่วนใหญ่ทำมือซ้ายหรือขวาก็สื่อความหมายเดียวกัน การกลับด้านข้อมูล
    จึงทำให้ได้ตัวอย่างเพิ่มฟรี ๆ เท่าตัว และโมเดลไม่ยึดติดว่าต้องใช้มือข้างไหน

    การกลับด้านทำสองอย่างพร้อมกัน: สลับเครื่องหมายแกน x และสลับช่องซ้าย-ขวา
    """
    mirrored = np.zeros_like(vector)

    for slot in range(len(HAND_SLOTS)):
        start = slot * PER_HAND_DIMS
        coords = vector[start:start + PER_HAND_DIMS - 1].reshape(NUM_LANDMARKS, COORDS_PER_LANDMARK)
        present = vector[start + PER_HAND_DIMS - 1]

        flipped = coords.copy()
        flipped[:, 0] *= -1.0        # กลับแกน x

        other = (slot + 1) % len(HAND_SLOTS)
        dest = other * PER_HAND_DIMS
        mirrored[dest:dest + PER_HAND_DIMS - 1] = flipped.flatten()
        mirrored[dest + PER_HAND_DIMS - 1] = present

    return mirrored


def feature_column_names() -> list[str]:
    """ชื่อคอลัมน์สำหรับหัวตาราง CSV เพื่อให้เปิดดูข้อมูลดิบแล้วเข้าใจได้"""
    names: list[str] = []
    for slot in HAND_SLOTS:
        for i in range(NUM_LANDMARKS):
            names += [f"{slot}_{i}_x", f"{slot}_{i}_y", f"{slot}_{i}_z"]
        names.append(f"{slot}_present")
    return names
