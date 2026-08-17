"""ตรวจสอบกล้องและการตรวจจับมือ — รันจบเองอัตโนมัติ ไม่มีหน้าต่างให้กดปิด

รัน:  run camtest

สคริปต์นี้ใช้ตอบคำถามว่า "กล้องใช้ได้ไหม" และ "MediaPipe มองเห็นมือเราหรือเปล่า"
ก่อนจะไปเสียเวลานั่งเก็บข้อมูลจริง ผลที่ได้จะบอกทั้งความเร็ว อัตราการตรวจเจอมือ
และบันทึกภาพตัวอย่างที่วาดโครงมือไว้ให้ดูด้วย
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

from src import drawing  # noqa: E402
from src.collect import open_camera  # noqa: E402
from src.console import enable_utf8_output  # noqa: E402
from src.hand_tracker import HandTracker  # noqa: E402

enable_utf8_output()


def main() -> int:
    parser = argparse.ArgumentParser(description="ตรวจสอบกล้องและการตรวจจับมือ")
    parser.add_argument("--seconds", type=float, default=10.0, help="ทดสอบนานกี่วินาที")
    parser.add_argument("--countdown", type=int, default=5, help="นับถอยหลังกี่วินาทีก่อนเริ่ม")
    args = parser.parse_args()

    print("=" * 62)
    print("ตรวจสอบกล้องและการตรวจจับมือ")
    print("=" * 62)

    try:
        cap = open_camera()
    except RuntimeError as exc:
        print(f"\n{exc}")
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\nเปิดกล้องหมายเลข {config.CAMERA_INDEX} สำเร็จ — ความละเอียด {width}x{height}")

    frames = 0
    frames_with_hands = 0
    max_hands_seen = 0
    best_frame = None
    best_count = 0
    next_tick = 1.0
    start = time.time()   # ค่าสำรอง เผื่อโหลดโมเดลไม่สำเร็จจนไม่ได้เข้าลูป

    try:
        print("กำลังโหลดโมเดลตรวจจับมือ ...", flush=True)
        with HandTracker() as tracker:
            # เริ่มจับเวลาหลังโหลดโมเดลเสร็จ ไม่งั้นเวลาที่ใช้โหลด (ราว 5 วินาที)
            # จะถูกนับรวมไปด้วย จนลูปหมดเวลาก่อนได้อ่านเฟรมแรก
            for remaining in range(args.countdown, 0, -1):
                print(f"  เตรียมยกมือ ... {remaining}", flush=True)
                cap.read()          # อ่านทิ้งไปเรื่อย ๆ เพื่อไม่ให้ภาพในบัฟเฟอร์ค้างเก่า
                time.sleep(1.0)

            print(f"\nเริ่มทดสอบ {args.seconds:.0f} วินาที — ยกมือค้างไว้ในกรอบภาพ\n", flush=True)
            start = time.time()

            while True:
                elapsed = time.time() - start
                if elapsed >= args.seconds:
                    break

                ok, frame = cap.read()
                if not ok:
                    print("อ่านภาพจากกล้องไม่ได้ — กล้องอาจถูกโปรแกรมอื่นใช้อยู่")
                    break

                if config.MIRROR_PREVIEW:
                    frame = cv2.flip(frame, 1)

                frames += 1
                hands = tracker.detect(frame, int(time.time() * 1000))

                if hands:
                    frames_with_hands += 1
                    max_hands_seen = max(max_hands_seen, len(hands))
                    if len(hands) >= best_count:
                        best_count = len(hands)
                        best_frame = drawing.draw_hands(frame.copy(), hands)

                if elapsed >= next_tick:
                    status = f"เห็นมือ {len(hands)} ข้าง" if hands else "ยังไม่เห็นมือ"
                    print(f"  {elapsed:4.1f} วินาที — {status}", flush=True)
                    next_tick += 1.0
    finally:
        cap.release()

    duration = time.time() - start
    fps = frames / duration if duration > 0 else 0.0
    detection_rate = frames_with_hands / frames if frames else 0.0

    print("\n" + "=" * 62)
    print("สรุปผล")
    print("=" * 62)
    print(f"  เฟรมทั้งหมด        : {frames} เฟรม ใน {duration:.1f} วินาที")
    print(f"  ความเร็ว           : {fps:.1f} FPS")
    print(f"  เฟรมที่ตรวจเจอมือ  : {frames_with_hands} ({detection_rate * 100:.0f}%)")
    print(f"  จำนวนมือมากที่สุด  : {max_hands_seen} ข้าง")

    if best_frame is not None:
        snapshot = config.DATA_DIR / "camtest_snapshot.jpg"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(snapshot), best_frame)
        print(f"\n  บันทึกภาพตัวอย่างที่วาดโครงมือไว้แล้ว -> {snapshot}")

    print()
    if frames == 0:
        print("กล้องเปิดได้แต่อ่านภาพไม่ออกเลย ลองเปลี่ยน CAMERA_INDEX ใน config.py")
        return 1
    if fps < 10:
        print("คำเตือน: FPS ต่ำกว่า 10 การใช้งานจริงจะกระตุก")
        print("         ลองลด FRAME_WIDTH/FRAME_HEIGHT ใน config.py เป็น 640x480")
    if frames_with_hands == 0:
        print("ไม่พบมือเลยตลอดการทดสอบ ตรวจสอบว่า:")
        print("  - ยกมือเข้ามาในกรอบภาพจริงหรือไม่")
        print("  - แสงสว่างพอไหม ถ้าย้อนแสงหรือมืดเกินไปจะตรวจไม่เจอ")
        return 1

    print("กล้องและการตรวจจับมือทำงานปกติ พร้อมเก็บข้อมูลแล้ว")
    print("ขั้นตอนถัดไป: run collect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
