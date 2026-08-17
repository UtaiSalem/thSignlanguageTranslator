"""แปลงจุด landmark ของมือให้เป็น "เวกเตอร์คุณลักษณะ" (feature vector)

นี่คือหัวใจของโปรเจค และเป็นเหตุผลที่วิธีนี้ใช้ข้อมูลน้อยกว่าการเทรนจากรูปภาพมาก

ปัญหาคือ พิกัดดิบจาก MediaPipe ขึ้นกับสิ่งที่ไม่เกี่ยวกับความหมายของท่ามือเลย:
    - มืออยู่ตรงไหนของเฟรม  (ตำแหน่ง)
    - มืออยู่ใกล้หรือไกลกล้อง (ขนาด)
เราจึงต้อง normalize ทิ้งสองอย่างนี้ออกไป เหลือไว้แค่ "รูปทรงของมือ" ล้วน ๆ:

    1. ย้ายจุดข้อมือ (landmark 0) ไปไว้ที่จุดกำเนิด  -> ตัดผลของตำแหน่ง
    2. หารด้วยระยะที่ไกลที่สุดจากข้อมือ              -> ตัดผลของขนาด

ผลลัพธ์: ทำท่าเดียวกันที่มุมไหนของจอ ใกล้หรือไกลแค่ไหน ก็ได้ตัวเลขชุดเดิม

แต่การ normalize แบบนี้มีราคาที่ต้องจ่าย: เมื่อแต่ละมือถูกเทียบกับข้อมือของตัวเอง
ความสัมพันธ์ระหว่างสองมือก็หายไปด้วย จึงต่อท้ายอีก 5 ค่าเพื่อเก็บส่วนนั้นกลับมา
(ดูหัวข้อ "ท่อนคู่มือ" ข้างล่าง) รวมเป็น 133 ค่า
"""

from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3          # x, y, z
PER_HAND_DIMS = NUM_LANDMARKS * COORDS_PER_LANDMARK + 1   # +1 = ธงบอกว่ามีมือข้างนี้ไหม
HAND_SLOTS = ("Left", "Right")   # ล็อกตำแหน่งในเวกเตอร์ให้คงที่เสมอ
HANDS_DIMS = PER_HAND_DIMS * len(HAND_SLOTS)              # = 128

# ---------------------------------------------------------------- ท่อนคู่มือ
# การ normalize ข้างบนเทียบกับข้อมือ "ของมือข้างนั้นเอง" ผลข้างเคียงคือความสัมพันธ์
# ระหว่างสองมือหายไปหมด: มือประกบกันกับมือแยกห่างกันจะได้ตัวเลขชุดเดียวกันเป๊ะ
# ซึ่งใช้ไม่ได้กับภาษามือไทยหลายคำที่ระยะห่างของสองมือเป็นตัวแยกความหมาย
#
# จึงต่อท้ายอีก 5 ค่าเพื่อเก็บ "ความสัมพันธ์ระหว่างสองมือ" ที่หายไป
PAIR_DIMS = 5
PAIR_START = HANDS_DIMS
FEATURE_DIMS = HANDS_DIMS + PAIR_DIMS                     # = 133

# จำนวนมิติของชุดข้อมูลรุ่นก่อนที่จะมีท่อนคู่มือ ใช้ตอนนำเข้าข้อมูลเก่า
LEGACY_FEATURE_DIMS = HANDS_DIMS


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """normalize จุด (21, 3) ของมือหนึ่งข้าง แล้วคืนเวกเตอร์แบน 63 ค่า"""
    normalized, _wrist, _scale = normalize_with_geometry(landmarks)
    return normalized


def normalize_with_geometry(landmarks: np.ndarray):
    """เหมือน normalize_landmarks แต่คืนตำแหน่งข้อมือและสเกลที่หารทิ้งไปด้วย

    สองค่าที่ปกติถูก "ทิ้ง" นี้คือสิ่งที่ท่อนคู่มือต้องใช้: ตำแหน่งข้อมือบอกว่า
    สองมืออยู่ห่างกันแค่ไหน และสเกลบอกว่ามือข้างไหนอยู่ใกล้กล้องกว่า
    """
    points = np.asarray(landmarks, dtype=np.float32).reshape(NUM_LANDMARKS, COORDS_PER_LANDMARK)

    # 1) ย้ายข้อมือไปที่จุดกำเนิด
    centered = points - points[0]

    # 2) ปรับสเกลด้วยระยะที่ไกลสุดจากข้อมือ
    scale = np.linalg.norm(centered, axis=1).max()
    if scale < 1e-6:          # กันหารด้วยศูนย์ ถ้าจุดซ้อนกันหมด
        scale = 1.0

    return (centered / scale).flatten(), points[0], scale


def build_feature_vector(hands) -> np.ndarray:
    """รวมมือที่ตรวจเจอ (0-2 ข้าง) เป็นเวกเตอร์ความยาวคงที่ 133 ค่า

    ต้องมีความยาวคงที่เพราะโมเดล machine learning รับ input ขนาดตายตัว
    ถ้าเจอมือข้างเดียว ช่องของอีกข้างจะเป็นศูนย์ทั้งหมด และธงบอกการมีอยู่ = 0

    โครงของเวกเตอร์:
        [  0..63 ]  มือ Left  : 63 ค่า normalize แล้ว + ธงบอกการมีอยู่
        [ 64..127]  มือ Right : เหมือนกัน
        [128..132]  ท่อนคู่มือ : offset x/y/z, สมดุลขนาด, ธงบอกว่าค่าใช้ได้
    """
    vector = np.zeros(FEATURE_DIMS, dtype=np.float32)
    wrists: dict[int, np.ndarray] = {}
    scales: dict[int, float] = {}

    for hand in hands:
        if hand.handedness not in HAND_SLOTS:
            continue
        slot = HAND_SLOTS.index(hand.handedness)
        start = slot * PER_HAND_DIMS
        normalized, wrist, scale = normalize_with_geometry(hand.landmarks)
        vector[start:start + PER_HAND_DIMS - 1] = normalized
        vector[start + PER_HAND_DIMS - 1] = 1.0   # ธง: มือข้างนี้ปรากฏอยู่
        wrists[slot] = wrist
        scales[slot] = scale

    if len(wrists) == len(HAND_SLOTS):
        _fill_pair_block(vector, wrists, scales)

    return vector


def _fill_pair_block(vector: np.ndarray, wrists: dict, scales: dict) -> None:
    """เติมท่อนคู่มือ — เรียกเฉพาะเมื่อเจอมือครบสองข้างเท่านั้น

    offset ถูกหารด้วยขนาดมือเฉลี่ย เพื่อให้ยังไม่ขึ้นกับระยะห่างจากกล้อง
    (ยืนไกลขึ้น ทั้ง offset และขนาดมือหดลงเท่ากัน อัตราส่วนจึงคงเดิม)

    สมดุลขนาดใช้สูตร (ขวา-ซ้าย)/(ขวา+ซ้าย) ซึ่งอยู่ในช่วง -1 ถึง 1 เสมอ
    เลือกแบบนี้แทนอัตราส่วนตรง ๆ เพราะไม่ระเบิดเมื่อตัวหารเล็ก และสลับเครื่องหมาย
    พอดีเมื่อกลับด้านซ้าย-ขวา (ดู mirror_feature_vector)
    """
    left_slot, right_slot = 0, 1
    left_scale = np.float32(scales[left_slot])
    right_scale = np.float32(scales[right_slot])

    mean_scale = (left_scale + right_scale) / np.float32(2.0)
    if mean_scale < 1e-6:
        mean_scale = np.float32(1.0)

    offset = (wrists[right_slot] - wrists[left_slot]) / mean_scale
    vector[PAIR_START:PAIR_START + 3] = offset

    total_scale = left_scale + right_scale
    if total_scale > 1e-6:
        vector[PAIR_START + 3] = (right_scale - left_scale) / total_scale

    vector[PAIR_START + 4] = 1.0   # ธง: ค่าในท่อนนี้วัดมาจริง ไม่ใช่ศูนย์เพราะไม่รู้


def build_canonical_feature_vector(hands, source_is_mirrored: bool) -> np.ndarray:
    """ประกอบเวกเตอร์ให้อยู่ใน "คอนเวนชันกลาง" ที่ทั้งคอมพิวเตอร์และมือถือใช้ร่วมกัน

    คอนเวนชันกลางคือ **ภาพดิบจากกล้อง (ยังไม่พลิกกระจก)** เหตุผลที่เลือกอันนี้
    เพราะฝั่งมือถือส่งภาพดิบเข้า MediaPipe อยู่แล้ว (พลิกกระจกด้วย CSS ตอนแสดงผล
    เท่านั้น) ฝั่งคอมพิวเตอร์จึงเป็นฝ่ายที่ต้องปรับให้ตรงกัน

    ทำไมฝั่งคอมพิวเตอร์ต้องปรับ: `src/collect.py` และ `src/translate.py` พลิกเฟรม
    ด้วย cv2.flip **ก่อน** ส่งเข้า MediaPipe (เพื่อให้พิกัดตรงกับภาพบนจอ วาดโครงมือ
    ได้ง่าย) ผลคือได้เวกเตอร์ของโลกกระจกเงา: แกน x กลับด้าน และ MediaPipe เรียก
    มือขวาจริงว่า "Left" ซึ่งเป็นภาพสะท้อนของคอนเวนชันกลางพอดี จึงแปลงกลับได้ด้วย
    mirror_feature_vector เพียงครั้งเดียว

    ถ้าไม่แปลง ข้อมูลจากสองฝั่งจะลงช่องมือสลับข้างกันและ x กลับด้าน ใช้ร่วมกันไม่ได้
    โดยไม่มี error ฟ้อง (ตรวจสอบด้วย `run parity` และ `python selftest.py`)
    """
    vector = build_feature_vector(hands)
    if source_is_mirrored:
        vector = mirror_feature_vector(vector)
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

    # ท่อนคู่มือกลับด้าน "ตรงข้าม" กับท่อนของแต่ละมือ ซึ่งดูขัดความรู้สึกแต่ถูกต้อง
    #
    # offset นิยามว่า (ข้อมือขวา - ข้อมือซ้าย) การกลับด้านทำให้เกิดสองอย่างพร้อมกัน
    # คือ x ของทุกจุดสลับเครื่องหมาย **และ** ช่องซ้าย-ขวาสลับกัน ซึ่งกลับทิศของ
    # ตัวลบเอง สองอย่างนี้หักล้างกันพอดีบนแกน x จึงคงเดิม ส่วนแกน y กับ z
    # ไม่ถูกกลับจากการพลิกภาพ เหลือแต่ผลจากการสลับตัวลบ จึงสลับเครื่องหมาย
    #
    #   offset'  = ข้อมือขวาใหม่ - ข้อมือซ้ายใหม่ = M(ข้อมือซ้าย) - M(ข้อมือขวา)
    #   offset'.x = (1-ซ้าย.x) - (1-ขวา.x) = ขวา.x - ซ้าย.x =  offset.x   (คงเดิม)
    #   offset'.y = ซ้าย.y - ขวา.y                          = -offset.y   (สลับ)
    mirrored[PAIR_START] = vector[PAIR_START]
    mirrored[PAIR_START + 1] = -vector[PAIR_START + 1]
    mirrored[PAIR_START + 2] = -vector[PAIR_START + 2]
    # สมดุลขนาด: มือขวากลายเป็นมือซ้าย ตัวตั้งกับตัวลบสลับกัน จึงสลับเครื่องหมาย
    mirrored[PAIR_START + 3] = -vector[PAIR_START + 3]
    mirrored[PAIR_START + 4] = vector[PAIR_START + 4]

    return mirrored


def upgrade_legacy_vector(values: np.ndarray) -> np.ndarray:
    """ขยายเวกเตอร์รุ่นเก่า 128 ค่า ให้เป็น 133 ค่า

    ท่อนคู่มือของข้อมูลเก่ากู้กลับไม่ได้จริง ๆ เพราะตำแหน่งข้อมือและขนาดมือถูก
    หารทิ้งไปตอนเก็บ จึงเติมศูนย์แล้วปล่อยธงเป็น 0 ซึ่งหมายถึง "ไม่รู้ค่า"
    ไม่ใช่ "สองมือซ้อนทับกันพอดี" โมเดลเรียนรู้ที่จะมองข้ามท่อนนี้เมื่อธงเป็น 0
    ได้เอง เหมือนที่มันเรียนรู้จากธงบอกการมีอยู่ของแต่ละมือ

    สำหรับตัวอย่างที่มีมือข้างเดียวหรือไม่มีมือเลย ค่าที่ถูกคือศูนย์และธงเป็น 0
    อยู่แล้ว ข้อมูลกลุ่มนี้จึงถูกกู้กลับได้ครบถ้วนไม่มีอะไรเสียหาย
    """
    upgraded = np.zeros(FEATURE_DIMS, dtype=np.float32)
    upgraded[:LEGACY_FEATURE_DIMS] = np.asarray(values, dtype=np.float32)
    return upgraded


def feature_column_names() -> list[str]:
    """ชื่อคอลัมน์สำหรับหัวตาราง CSV เพื่อให้เปิดดูข้อมูลดิบแล้วเข้าใจได้"""
    names: list[str] = []
    for slot in HAND_SLOTS:
        for i in range(NUM_LANDMARKS):
            names += [f"{slot}_{i}_x", f"{slot}_{i}_y", f"{slot}_{i}_z"]
        names.append(f"{slot}_present")
    names += [
        "Pair_offset_x",
        "Pair_offset_y",
        "Pair_offset_z",
        "Pair_scale_balance",
        "Pair_present",
    ]
    return names
