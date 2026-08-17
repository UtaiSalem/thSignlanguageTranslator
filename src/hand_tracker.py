"""ห่อ MediaPipe HandLandmarker ให้เรียกใช้ง่าย

หมายเหตุสำคัญ: mediapipe ตั้งแต่รุ่น 0.10.3x เป็นต้นไป **ตัด API เดิม**
(`mp.solutions.hands`) ออกไปแล้ว โค้ดสอนเก่า ๆ ที่ใช้ cvzone จึงรันไม่ได้
ไฟล์นี้ใช้ Tasks API รุ่นใหม่ ซึ่งต้องโหลดไฟล์โมเดล .task มาไว้ในเครื่อง
"""

from __future__ import annotations

import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


# เส้นเชื่อมระหว่างจุด 21 จุดบนมือ ใช้ตอนวาดโครงมือ
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # นิ้วโป้ง
    (0, 5), (5, 6), (6, 7), (7, 8),            # นิ้วชี้
    (5, 9), (9, 10), (10, 11), (11, 12),       # นิ้วกลาง
    (9, 13), (13, 14), (14, 15), (15, 16),     # นิ้วนาง
    (13, 17), (17, 18), (18, 19), (19, 20),    # นิ้วก้อย
    (0, 17),                                   # ฝ่ามือ
]


@dataclass
class Hand:
    """ผลตรวจจับมือหนึ่งข้าง"""

    landmarks: np.ndarray   # (21, 3) พิกัด x, y, z แบบ normalize 0-1
    handedness: str         # "Left" หรือ "Right"
    score: float

    def bbox(self, frame_w: int, frame_h: int, padding: int = 20):
        """คืนกรอบสี่เหลี่ยมล้อมมือเป็นพิกเซล (x1, y1, x2, y2)"""
        xs = self.landmarks[:, 0] * frame_w
        ys = self.landmarks[:, 1] * frame_h
        x1 = max(int(xs.min()) - padding, 0)
        y1 = max(int(ys.min()) - padding, 0)
        x2 = min(int(xs.max()) + padding, frame_w - 1)
        y2 = min(int(ys.max()) + padding, frame_h - 1)
        return x1, y1, x2, y2


def ensure_model_file() -> Path:
    """ดาวน์โหลดไฟล์โมเดลตรวจจับมือถ้ายังไม่มีในเครื่อง"""
    path = config.HAND_LANDMARKER_TASK
    if path.exists() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"กำลังดาวน์โหลดโมเดลตรวจจับมือ -> {path}")
    urllib.request.urlretrieve(config.HAND_LANDMARKER_URL, path)
    print(f"ดาวน์โหลดเสร็จ ({path.stat().st_size / 1e6:.1f} MB)")
    return path


class HandTracker:
    """ตัวตรวจจับมือแบบสตรีมวิดีโอ ใช้เป็น context manager ได้

    ตัวอย่าง:
        with HandTracker() as tracker:
            hands = tracker.detect(frame_bgr, timestamp_ms)
    """

    def __init__(self, max_hands: int | None = None):
        model_path = ensure_model_file()
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands if max_hands is not None else config.MAX_HANDS,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[Hand]:
        """ตรวจจับมือจากเฟรม BGR (รูปแบบมาตรฐานของ OpenCV)

        Tasks API แบบ VIDEO บังคับให้ timestamp ต้องเพิ่มขึ้นเสมอ
        ถ้าเผลอส่งค่าย้อนหลังมาจะ error จึงกันไว้ที่นี่เลย
        """
        if timestamp_ms <= self._last_timestamp:
            timestamp_ms = self._last_timestamp + 1
        self._last_timestamp = timestamp_ms

        rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        hands: list[Hand] = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            points = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
            category = handedness[0]
            hands.append(Hand(points, category.category_name, float(category.score)))
        return hands

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
