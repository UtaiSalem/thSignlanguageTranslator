"""ขั้นตอนที่ 3 — แปลภาษามือเป็นข้อความแบบเรียลไทม์

รัน:  python -m src.translate

ปุ่มควบคุม
    SPACE     เว้นวรรค (คั่นคำด้วยตัวเอง)
    BACKSPACE ลบคำล่าสุด
    C         ล้างข้อความทั้งหมด
    S         บันทึกข้อความลงไฟล์
    Q / ESC   ออกจากโปรแกรม

หัวใจของไฟล์นี้ไม่ใช่แค่การทายท่ามือ แต่คือการเปลี่ยน "ผลทายรายเฟรม"
ให้กลายเป็น "ประโยค" ที่อ่านรู้เรื่อง กล้องส่งภาพมา 30 เฟรมต่อวินาที
ถ้ารับทุกผลที่ทายได้ ข้อความจะกลายเป็น "สวัสดีสวัสดีสวัสดี..." ทันที
จึงต้องมีกลไกสามชั้นคอยกรอง: ความมั่นใจ, ความนิ่ง, และช่วงพักหลังรับคำ
"""

from __future__ import annotations

import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

from . import drawing, thai_text  # noqa: E402
from .collect import open_camera  # noqa: E402
from .features import build_feature_vector  # noqa: E402
from .hand_tracker import HandTracker  # noqa: E402


class SentenceBuilder:
    """รวมผลทายรายเฟรมให้กลายเป็นประโยค

    ตัวกรองสามชั้น:
        1. ความมั่นใจ — ผลที่โมเดลไม่มั่นใจพอ ถือว่ายังไม่ใช่ท่าที่ตั้งใจทำ
        2. ความนิ่ง   — ต้องทายคำเดิมติดกันหลายเฟรม จึงกันช่วงมือเปลี่ยนท่าออกไปได้
        3. ช่วงพัก    — หลังรับคำแล้วจะไม่รับคำใหม่ทันที กันคำเดิมซ้ำรัว ๆ
    """

    def __init__(self, labels: list[str] | None = None):
        self.labels = labels or []      # คลังคำของโมเดล ใช้ตอนวาดอันดับการทาย
        self.words: list[str] = []
        self._recent: deque[str] = deque(maxlen=config.STABLE_FRAMES_REQUIRED)
        self._accepted_at = 0.0
        self._last_accepted: str | None = None

    def update(self, label: str | None, confidence: float) -> str | None:
        """ป้อนผลทายของเฟรมนี้ คืนคำที่เพิ่งถูกรับเข้าประโยค (ถ้ามี)"""
        if label is None or confidence < config.CONFIDENCE_THRESHOLD:
            self._recent.clear()
            return None

        self._recent.append(label)

        # ชั้นที่ 2: ต้องนิ่งครบจำนวนเฟรมที่กำหนด และต้องเป็นคำเดียวกันทั้งหมด
        if len(self._recent) < self._recent.maxlen or len(set(self._recent)) != 1:
            return None

        # ชั้นที่ 3: ยังอยู่ในช่วงพักของคำเดิม
        now = time.time()
        if label == self._last_accepted and now - self._accepted_at < config.COOLDOWN_SEC:
            return None

        self.words.append(label)
        self._last_accepted = label
        self._accepted_at = now
        self._recent.clear()
        return label

    @property
    def progress(self) -> float:
        """สัดส่วนความนิ่งที่สะสมไว้ 0-1 ใช้วาดเป็นวงแหวนให้ผู้ใช้เห็นว่าใกล้ได้แล้ว"""
        return len(self._recent) / self._recent.maxlen

    @property
    def text(self) -> str:
        return " ".join(self.words)

    def add_space(self):
        if self.words and self.words[-1] != "|":
            self.words.append("|")

    def backspace(self):
        if self.words:
            self.words.pop()
        self._recent.clear()
        self._last_accepted = None

    def clear(self):
        self.words.clear()
        self._recent.clear()
        self._last_accepted = None


def load_classifier():
    """โหลดโมเดลที่เทรนไว้ พร้อมข้อความบอกทางถ้ายังไม่ได้เทรน"""
    if not config.CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"ยังไม่มีไฟล์โมเดล {config.CLASSIFIER_PATH}\n"
            f"ให้เก็บข้อมูลด้วย `python -m src.collect` แล้วเทรนด้วย `python -m src.train` ก่อน"
        )
    bundle = joblib.load(config.CLASSIFIER_PATH)
    # แปลงเป็น str ปกติ เพราะ scikit-learn คืนค่าเป็น numpy.str_ ซึ่งพาไปโผล่
    # ในข้อความที่บันทึกลงไฟล์ในรูปแบบแปลก ๆ ได้
    labels = [str(label) for label in bundle["labels"]]
    print(
        f"โหลดโมเดล {bundle['model_kind']} สำเร็จ "
        f"({len(labels)} คำ, ความแม่นยำตอนทดสอบ {bundle['accuracy'] * 100:.1f}%)"
    )
    return bundle["model"], labels


def predict(model, labels: list[str], hands):
    """ทายท่ามือของเฟรมนี้ คืน (คำที่ทาย, ความมั่นใจ, ความน่าจะเป็นทุกคำ)"""
    if not hands:
        return None, 0.0, None

    vector = build_feature_vector(hands).reshape(1, -1)
    probabilities = model.predict_proba(vector)[0]
    best = int(np.argmax(probabilities))
    return labels[best], float(probabilities[best]), probabilities


def draw_top_predictions(frame, labels, probabilities, top_k: int = 3):
    """แสดงคำที่โมเดลคิดว่าน่าจะใช่ที่สุด 3 อันดับ พร้อมแถบความมั่นใจ

    ส่วนนี้มีไว้เพื่อการเรียนรู้โดยเฉพาะ: ทำให้เห็นว่าโมเดล "ลังเล" ระหว่างคำไหน
    ซึ่งช่วยชี้ว่าควรไปเก็บข้อมูลของคำคู่ไหนเพิ่ม
    """
    if probabilities is None:
        return frame

    order = np.argsort(probabilities)[::-1][:top_k]
    x, y = frame.shape[1] - 330, 170
    drawing.draw_panel(frame, (x - 15, y - 15), (frame.shape[1] - 15, y + 40 * top_k + 5))

    for rank, idx in enumerate(order):
        row_y = y + rank * 40
        color = (0, 255, 0) if rank == 0 else (190, 190, 190)
        frame = thai_text.draw_text(frame, labels[idx], (x, row_y), 26, color)
        drawing.draw_progress_bar(
            frame, (x + 150, row_y + 10), 130, 12, float(probabilities[idx]), color
        )
    return frame


def render_overlay(frame, hands, label, confidence, probabilities, builder, fps=None):
    """วาดหน้าจอทั้งหมดลงบนเฟรม แล้วคืนเฟรมใหม่

    แยกออกมาจากลูปหลักเพื่อให้ทดสอบการแสดงผลได้โดยไม่ต้องเปิดกล้อง
    """
    drawing.draw_hands(frame, hands)
    height, width = frame.shape[:2]

    # ---- แผงบน: คำที่กำลังทาย ----
    drawing.draw_panel(frame, (0, 0), (width, 150))
    if label is not None and confidence >= config.CONFIDENCE_THRESHOLD:
        frame = thai_text.draw_text(frame, label, (20, 10), 60, (0, 255, 200))
        frame = thai_text.draw_text(
            frame, f"ความมั่นใจ {confidence * 100:.0f}%", (20, 88), 26, (255, 255, 255)
        )
        drawing.draw_progress_bar(frame, (20, 124), 280, 12, builder.progress)
        frame = thai_text.draw_text(frame, "ค้างท่าไว้...", (315, 112), 20, (200, 200, 200))
    elif hands:
        frame = thai_text.draw_text(frame, "ไม่แน่ใจ", (20, 20), 48, (0, 140, 255))
        frame = thai_text.draw_text(
            frame, "ปรับท่าให้ชัดขึ้น หรือเก็บข้อมูลคำนี้เพิ่ม", (20, 92), 24, (200, 200, 200)
        )
    else:
        frame = thai_text.draw_text(frame, "ยกมือเข้ามาในกรอบ", (20, 30), 44, (180, 180, 180))

    frame = draw_top_predictions(frame, builder.labels, probabilities)

    # ---- แผงล่าง: ประโยคที่แปลได้ ----
    drawing.draw_panel(frame, (0, height - 120), (width, height))
    sentence = builder.text.replace("|", "   ")
    frame = thai_text.draw_text(frame, "ข้อความที่แปลได้", (20, height - 112), 22, (150, 200, 255))
    frame = thai_text.draw_text(frame, sentence or "—", (20, height - 82), 40, (255, 255, 255))
    frame = thai_text.draw_text(
        frame,
        "SPACE=เว้นวรรค  BACKSPACE=ลบคำ  C=ล้าง  S=บันทึก  Q=ออก",
        (20, height - 30), 20, (200, 200, 200),
    )

    if fps is not None:
        cv2.putText(
            frame, f"{fps:4.1f} FPS", (width - 120, height - 132),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1,
        )

    return frame


def save_transcript(text: str) -> Path | None:
    """บันทึกข้อความที่แปลได้ลงไฟล์ พร้อมประทับเวลา"""
    if not text.strip():
        print("ยังไม่มีข้อความให้บันทึก")
        return None

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = config.DATA_DIR / "transcripts.txt"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {text}\n")
    print(f"บันทึกข้อความแล้ว -> {path}")
    return path


def main() -> int:
    try:
        model, labels = load_classifier()
    except FileNotFoundError as exc:
        print(f"\n{exc}")
        return 1
    builder = SentenceBuilder(labels)
    flash_until = 0.0
    fps_times: deque[float] = deque(maxlen=30)

    cap = open_camera()
    try:
        with HandTracker() as tracker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("อ่านภาพจากกล้องไม่ได้")
                    break

                if config.MIRROR_PREVIEW:
                    frame = cv2.flip(frame, 1)

                now = time.time()
                fps_times.append(now)

                hands = tracker.detect(frame, int(now * 1000))
                label, confidence, probabilities = predict(model, labels, hands)
                if builder.update(label, confidence):
                    flash_until = now + 0.25

                fps = None
                if len(fps_times) > 1 and fps_times[-1] > fps_times[0]:
                    fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0])

                frame = render_overlay(
                    frame, hands, label, confidence, probabilities, builder, fps
                )

                if now < flash_until:
                    height, width = frame.shape[:2]
                    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 255, 0), 8)

                cv2.imshow("แปลภาษามือไทยเป็นข้อความ - Sign Language Translator", frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break
                elif key == ord(" "):
                    builder.add_space()
                elif key == 8:            # BACKSPACE
                    builder.backspace()
                elif key == ord("c"):
                    builder.clear()
                elif key == ord("s"):
                    save_transcript(builder.text.replace("|", " "))
    finally:
        cap.release()
        cv2.destroyAllWindows()

    final_text = builder.text.replace("|", " ").strip()
    print(f"\nข้อความสุดท้าย: {final_text or '(ว่าง)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
