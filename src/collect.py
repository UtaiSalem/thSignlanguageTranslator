"""ขั้นตอนที่ 1 — เก็บข้อมูลท่ามือจากกล้อง

รัน:  python -m src.collect

ปุ่มควบคุม
    SPACE   เก็บ 1 ตัวอย่าง
    C       เปิด/ปิดโหมดเก็บต่อเนื่อง (กดค้างท่าไว้แล้วขยับมือเล็กน้อย)
    N / P   เปลี่ยนไปคำถัดไป / คำก่อนหน้า
    D       ลบตัวอย่างล่าสุดของคำปัจจุบัน
    Q / ESC ออกจากโปรแกรม

เคล็ดลับให้โมเดลแม่น: ระหว่างเก็บข้อมูลให้ค่อย ๆ หมุนข้อมือ ขยับเข้า-ออกจากกล้อง
และเปลี่ยนตำแหน่งมือในเฟรม เพื่อให้โมเดลเห็นท่าเดียวกันในหลาย ๆ มุม
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

from . import dataset, drawing, thai_text  # noqa: E402
from .console import enable_utf8_output  # noqa: E402
from .features import build_canonical_feature_vector  # noqa: E402
from .hand_tracker import HandTracker  # noqa: E402

enable_utf8_output()


def open_camera() -> cv2.VideoCapture:
    """เปิดกล้อง — บน Windows ใช้ backend DSHOW จะเปิดไวกว่าค่าเริ่มต้นมาก"""
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(config.CAMERA_INDEX, backend)
    if not cap.isOpened():
        raise RuntimeError(
            f"เปิดกล้องหมายเลข {config.CAMERA_INDEX} ไม่ได้\n"
            f"ลองเปลี่ยนค่า CAMERA_INDEX ใน config.py เป็น 1 หรือ 2 "
            f"และตรวจว่าไม่มีโปรแกรมอื่นใช้กล้องอยู่"
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap


def draw_hud(frame, label: str, index: int, total: int, counts, burst: bool, hand_count: int):
    """วาดแผงข้อมูลด้านบนของหน้าจอ"""
    collected = counts.get(label, 0)
    target = config.SAMPLES_PER_CLASS_TARGET

    drawing.draw_panel(frame, (0, 0), (frame.shape[1], 150))

    frame = thai_text.draw_text(
        frame, f"กำลังเก็บคำว่า: {label}", (20, 12), 42, (0, 255, 200)
    )
    frame = thai_text.draw_text(
        frame,
        f"คำที่ {index + 1}/{total}   เก็บแล้ว {collected}/{target} ตัวอย่าง   "
        f"รวมทั้งชุด {sum(counts.values())}",
        (20, 66), 24, (255, 255, 255),
    )

    status = "โหมดต่อเนื่อง: เปิด" if burst else "โหมดต่อเนื่อง: ปิด (กด C)"
    status_color = (0, 255, 0) if burst else (160, 160, 160)
    frame = thai_text.draw_text(frame, status, (20, 100), 24, status_color)

    hand_text = f"เห็นมือ {hand_count} ข้าง" if hand_count else "ไม่เห็นมือ — ยกมือเข้ากรอบ"
    hand_color = (0, 255, 0) if hand_count else (0, 80, 255)
    frame = thai_text.draw_text(frame, hand_text, (frame.shape[1] - 320, 100), 24, hand_color)

    drawing.draw_progress_bar(frame, (frame.shape[1] - 320, 60), 280, 14, collected / target)

    frame = thai_text.draw_text(
        frame,
        "SPACE=เก็บ  C=ต่อเนื่อง  N/P=เปลี่ยนคำ  D=ลบล่าสุด  Q=ออก",
        (20, frame.shape[0] - 40), 22, (230, 230, 230), (0, 0, 0),
    )
    return frame


def main() -> int:
    labels = config.ACTIVE_LABELS
    if not labels:
        print("ยังไม่ได้กำหนดคำใน ACTIVE_LABELS ที่ config.py")
        return 1

    counts = dataset.count_samples(config.DATASET_CSV)
    index = 0
    burst = False
    last_capture = 0.0
    flash_until = 0.0

    print(f"เก็บข้อมูลลงไฟล์: {config.DATASET_CSV}")
    print(f"จำนวนคำทั้งหมด: {len(labels)}")

    cap = open_camera()
    try:
        with HandTracker() as tracker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("อ่านภาพจากกล้องไม่ได้")
                    break

                # พลิกภาพก่อนตรวจจับ เพื่อให้พิกัดที่ได้ตรงกับภาพที่เห็นบนจอ
                # ผลข้างเคียง: MediaPipe จะเรียกมือขวาจริงว่า "Left" และแกน x กลับด้าน
                # จึงต้องแปลงเวกเตอร์กลับเป็นคอนเวนชันกลางด้วย
                # build_canonical_feature_vector ก่อนบันทึก ไม่งั้นข้อมูลชุดนี้จะ
                # ใช้ร่วมกับข้อมูลจากแอปมือถือไม่ได้ (ดู src/features.py)
                if config.MIRROR_PREVIEW:
                    frame = cv2.flip(frame, 1)

                hands = tracker.detect(frame, int(time.time() * 1000))
                drawing.draw_hands(frame, hands)

                label = labels[index]
                now = time.time()

                # เก็บอัตโนมัติในโหมดต่อเนื่อง
                if burst and hands and now - last_capture >= config.BURST_INTERVAL_SEC:
                    dataset.append_sample(
                        config.DATASET_CSV,
                        label,
                        build_canonical_feature_vector(hands, config.MIRROR_PREVIEW),
                    )
                    counts[label] += 1
                    last_capture = now
                    flash_until = now + 0.06

                frame = draw_hud(frame, label, index, len(labels), counts, burst, len(hands))

                if now < flash_until:   # กะพริบขอบเขียวบอกว่าเพิ่งเก็บไป 1 ตัวอย่าง
                    cv2.rectangle(
                        frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 255, 0), 8
                    )

                cv2.imshow("เก็บข้อมูลภาษามือ - Data Collection", frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break
                elif key == ord(" "):
                    if hands:
                        dataset.append_sample(
                            config.DATASET_CSV,
                            label,
                            build_canonical_feature_vector(hands, config.MIRROR_PREVIEW),
                        )
                        counts[label] += 1
                        flash_until = now + 0.12
                        print(f"[{label}] เก็บแล้ว {counts[label]} ตัวอย่าง")
                    else:
                        print("ไม่เห็นมือในเฟรม — ยังไม่เก็บ")
                elif key == ord("c"):
                    burst = not burst
                    last_capture = 0.0
                elif key == ord("n"):
                    index = (index + 1) % len(labels)
                    burst = False
                elif key == ord("p"):
                    index = (index - 1) % len(labels)
                    burst = False
                elif key == ord("d"):
                    if dataset.drop_last_sample(config.DATASET_CSV, label):
                        counts[label] = max(counts[label] - 1, 0)
                        print(f"ลบตัวอย่างล่าสุดของ [{label}] แล้ว เหลือ {counts[label]}")
                    else:
                        print(f"ไม่มีตัวอย่างของ [{label}] ให้ลบ")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\nสรุปจำนวนตัวอย่างที่เก็บได้")
    for label in labels:
        print(f"  {label:<12} {counts.get(label, 0)}")
    print(f"\nรวม {sum(counts.values())} ตัวอย่าง -> {config.DATASET_CSV}")
    print("ขั้นตอนถัดไป: python -m src.train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
