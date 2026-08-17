"""ฟังก์ชันวาดภาพซ้อนบนเฟรม ใช้ร่วมกันทั้งตอนเก็บข้อมูลและตอนแปลผล"""

from __future__ import annotations

import cv2
import numpy as np

from .hand_tracker import HAND_CONNECTIONS

# สี BGR
COLOR_BONE = (200, 200, 200)
COLOR_JOINT = (0, 220, 255)
COLOR_BOX = (0, 200, 0)


def draw_hands(frame: np.ndarray, hands, show_box: bool = True) -> np.ndarray:
    """วาดโครงกระดูกมือ 21 จุดและกรอบล้อมมือลงบนเฟรม (แก้ไขเฟรมเดิมโดยตรง)"""
    h, w = frame.shape[:2]

    for hand in hands:
        points = [(int(x * w), int(y * h)) for x, y, _ in hand.landmarks]

        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], COLOR_BONE, 2)
        for point in points:
            cv2.circle(frame, point, 4, COLOR_JOINT, cv2.FILLED)

        if show_box:
            x1, y1, x2, y2 = hand.bbox(w, h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)

    return frame


def draw_panel(
    frame: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.55,
) -> np.ndarray:
    """วาดแผงพื้นหลังโปร่งแสง เพื่อให้ตัวหนังสืออ่านออกไม่ว่าฉากหลังจะเป็นอะไร"""
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_progress_bar(
    frame: np.ndarray,
    top_left: tuple[int, int],
    width: int,
    height: int,
    ratio: float,
    color: tuple[int, int, int] = (0, 200, 0),
) -> np.ndarray:
    """แถบความคืบหน้า ใช้บอกว่าเก็บข้อมูลคำปัจจุบันไปกี่เปอร์เซ็นต์แล้ว"""
    x, y = top_left
    ratio = float(np.clip(ratio, 0.0, 1.0))
    cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 80, 80), cv2.FILLED)
    if ratio > 0:
        cv2.rectangle(frame, (x, y), (x + int(width * ratio), y + height), color, cv2.FILLED)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (220, 220, 220), 1)
    return frame
