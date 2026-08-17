"""ตรวจสอบว่าติดตั้งทุกอย่างถูกต้องหรือยัง — รันได้โดยไม่ต้องใช้กล้อง

รัน:  python selftest.py

สคริปต์นี้ไล่ทดสอบทุกชิ้นส่วนของระบบด้วยข้อมูลจำลอง ถ้าผ่านครบทุกข้อ
แปลว่าสภาพแวดล้อมพร้อมใช้งาน เหลือแค่เสียบกล้องแล้วเริ่มเก็บข้อมูลจริง
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str):
    """ตัวช่วยรันการทดสอบหนึ่งข้อ แล้วเก็บผลไว้สรุปตอนท้าย"""
    def decorator(func):
        try:
            detail = func()
            PASSED.append(f"{name}" + (f" — {detail}" if detail else ""))
        except Exception as exc:  # noqa: BLE001
            FAILED.append((name, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
        return func
    return decorator


class FakeHand:
    """มือจำลอง ใช้แทนผลจาก MediaPipe เพื่อทดสอบโดยไม่ต้องมีกล้อง"""

    def __init__(self, landmarks, handedness):
        self.landmarks = landmarks
        self.handedness = handedness
        self.score = 0.99


def fake_landmarks(seed: int, jitter: float = 0.0) -> np.ndarray:
    """สร้างจุดมือ 21 จุดแบบสุ่มแต่ทำซ้ำได้ ใช้แทน 'ท่ามือ' หนึ่งท่า"""
    rng = np.random.default_rng(seed)
    base = rng.random((21, 3)).astype(np.float32) * 0.3 + 0.35
    if jitter:
        base += np.random.default_rng(seed * 7919).normal(0, jitter, base.shape).astype(np.float32)
    return base


def main() -> int:
    print("=" * 62)
    print("ตรวจสอบสภาพแวดล้อมของโปรเจคแปลภาษามือไทย")
    print("=" * 62)
    print()

    @check("นำเข้าไลบรารีหลัก")
    def _():
        import cv2, joblib, mediapipe, sklearn  # noqa: F401
        from PIL import Image  # noqa: F401
        return (
            f"python {sys.version_info.major}.{sys.version_info.minor}, "
            f"opencv {cv2.__version__}, mediapipe {mediapipe.__version__}, "
            f"sklearn {sklearn.__version__}"
        )

    @check("มีไฟล์โมเดลตรวจจับมือ")
    def _():
        from src.hand_tracker import ensure_model_file
        path = ensure_model_file()
        return f"{path.name} ({path.stat().st_size / 1e6:.1f} MB)"

    @check("สร้างตัวตรวจจับมือได้")
    def _():
        from src.hand_tracker import HandTracker
        with HandTracker() as tracker:
            hands = tracker.detect(np.zeros((480, 640, 3), dtype=np.uint8), 0)
        return f"ตรวจภาพเปล่าแล้วพบมือ {len(hands)} ข้าง (ถูกต้องคือ 0)"

    @check("normalize จุด landmark ตัดผลของตำแหน่งและขนาดได้จริง")
    def _():
        from src.features import normalize_landmarks
        points = fake_landmarks(1)
        moved = points + np.array([0.2, -0.1, 0.05], dtype=np.float32)   # ย้ายตำแหน่ง
        scaled = points * 2.5                                            # เปลี่ยนขนาด
        a, b, c = (normalize_landmarks(p) for p in (points, moved, scaled))
        assert np.allclose(a, b, atol=1e-5), "ย้ายตำแหน่งแล้วค่าไม่เท่าเดิม"
        assert np.allclose(a, c, atol=1e-5), "เปลี่ยนขนาดแล้วค่าไม่เท่าเดิม"
        return "ย้ายมือและซูมเข้าออกแล้วได้ค่าเดิม"

    @check("ประกอบ feature vector ขนาดคงที่")
    def _():
        from src.features import FEATURE_DIMS, build_feature_vector
        one = build_feature_vector([FakeHand(fake_landmarks(2), "Right")])
        two = build_feature_vector(
            [FakeHand(fake_landmarks(2), "Right"), FakeHand(fake_landmarks(3), "Left")]
        )
        none = build_feature_vector([])
        assert one.shape == two.shape == none.shape == (FEATURE_DIMS,)
        assert none.sum() == 0, "ไม่มีมือแต่เวกเตอร์ไม่เป็นศูนย์"
        assert one[63] == 0.0 and one[127] == 1.0, "ธงบอกการมีอยู่ของมือผิดช่อง"
        return f"{FEATURE_DIMS} มิติ ทั้งกรณี 0/1/2 มือ"

    @check("กลับด้านซ้าย-ขวาแล้วกลับซ้ำได้ค่าเดิม")
    def _():
        from src.features import build_feature_vector, mirror_feature_vector
        original = build_feature_vector([FakeHand(fake_landmarks(4), "Right")])
        twice = mirror_feature_vector(mirror_feature_vector(original))
        assert np.allclose(original, twice, atol=1e-6), "กลับด้านสองครั้งแล้วไม่กลับมาเหมือนเดิม"
        mirrored = mirror_feature_vector(original)
        assert mirrored[63] == 1.0 and mirrored[127] == 0.0, "สลับช่องซ้าย-ขวาไม่ถูก"
        return "สลับช่องและกลับแกน x ถูกต้อง"

    @check("วาดข้อความภาษาไทยลงบนภาพได้")
    def _():
        from src import thai_text
        frame = np.zeros((200, 800, 3), dtype=np.uint8)
        out = thai_text.draw_text(frame, "สวัสดี ผู้ใช้ ก-ฮ", (20, 40), 48, (0, 255, 0))
        assert out.shape == frame.shape
        assert (out > 0).any(), "วาดแล้วภาพยังดำสนิท แปลว่าหาฟอนต์ไทยไม่เจอ"
        return "เรนเดอร์ด้วยฟอนต์ไทยสำเร็จ"

    @check("เขียนและอ่านไฟล์ชุดข้อมูล CSV")
    def _():
        from src import dataset
        from src.features import build_feature_vector
        tmp = Path(tempfile.mkdtemp()) / "test.csv"
        try:
            for i in range(6):
                vector = build_feature_vector([FakeHand(fake_landmarks(i % 3), "Right")])
                dataset.append_sample(tmp, f"คำที่{i % 3}", vector)
            X, y = dataset.load_dataset(tmp)
            assert X.shape == (6, 128) and len(set(y)) == 3
            assert dataset.count_samples(tmp)["คำที่0"] == 2
            assert dataset.drop_last_sample(tmp, "คำที่0") is True
            assert dataset.count_samples(tmp)["คำที่0"] == 1
            return "บันทึก อ่านกลับ นับ และลบตัวอย่าง ครบถ้วน"
        finally:
            shutil.rmtree(tmp.parent, ignore_errors=True)

    @check("เทรนโมเดลและทำนายผลได้")
    def _():
        from sklearn.metrics import accuracy_score
        from src.features import build_feature_vector
        from src.train import augment_with_mirror, build_model

        # สร้างข้อมูลจำลอง 4 คำ คำละ 40 ตัวอย่าง (ท่าเดิมแต่สั่นเล็กน้อย)
        X, y = [], []
        for cls in range(4):
            for rep in range(40):
                points = fake_landmarks(100 + cls) + np.random.default_rng(
                    cls * 1000 + rep
                ).normal(0, 0.004, (21, 3)).astype(np.float32)
                X.append(build_feature_vector([FakeHand(points, "Right")]))
                y.append(f"คำ{cls}")
        X, y = np.array(X), np.array(y)

        X_aug, y_aug = augment_with_mirror(X, y)
        assert len(X_aug) == 2 * len(X)

        model = build_model("mlp")
        model.fit(X_aug, y_aug)
        accuracy = accuracy_score(y, model.predict(X))
        assert accuracy > 0.9, f"ความแม่นยำบนข้อมูลจำลองต่ำผิดปกติ: {accuracy:.2f}"

        probabilities = model.predict_proba(X[:1])[0]
        assert abs(probabilities.sum() - 1.0) < 1e-5
        return f"4 คำ, ความแม่นยำ {accuracy * 100:.1f}%, predict_proba ใช้งานได้"

    @check("ตัวรวมประโยคกรองผลรายเฟรมถูกต้อง")
    def _():
        from src.translate import SentenceBuilder
        builder = SentenceBuilder()

        # ความมั่นใจต่ำ ต้องไม่ถูกรับ
        for _ in range(config.STABLE_FRAMES_REQUIRED * 2):
            assert builder.update("สวัสดี", 0.10) is None
        assert builder.words == [], "คำที่ไม่มั่นใจหลุดเข้าประโยค"

        # มั่นใจแต่ท่าไม่นิ่ง (สลับไปมา) ต้องไม่ถูกรับ
        builder = SentenceBuilder()
        for i in range(config.STABLE_FRAMES_REQUIRED * 2):
            builder.update("สวัสดี" if i % 2 else "ขอบคุณ", 0.99)
        assert builder.words == [], "คำที่ยังไม่นิ่งหลุดเข้าประโยค"

        # มั่นใจและนิ่งครบ ต้องถูกรับพอดี 1 ครั้ง
        builder = SentenceBuilder()
        accepted = [builder.update("สวัสดี", 0.99) for _ in range(config.STABLE_FRAMES_REQUIRED)]
        assert accepted.count("สวัสดี") == 1, f"ควรรับพอดี 1 ครั้ง แต่ได้ {accepted}"
        assert accepted[:-1] == [None] * (config.STABLE_FRAMES_REQUIRED - 1), (
            "รับคำเร็วเกินไป ยังนิ่งไม่ครบจำนวนเฟรมที่กำหนด"
        )
        assert builder.words == ["สวัสดี"]

        # ช่วงพัก: ทำท่าเดิมต่อทันที ต้องยังไม่ถูกรับซ้ำ
        for _ in range(config.STABLE_FRAMES_REQUIRED * 3):
            builder.update("สวัสดี", 0.99)
        assert builder.words == ["สวัสดี"], f"คำซ้ำหลุดเข้ามา: {builder.words}"

        builder.backspace()
        assert builder.words == []
        return "กรองครบทั้งความมั่นใจ ความนิ่ง และช่วงพัก"

    @check("ค่าตั้งใน config.py สมเหตุสมผล")
    def _():
        assert config.ACTIVE_LABELS, "ACTIVE_LABELS ว่างเปล่า"
        assert len(set(config.ACTIVE_LABELS)) == len(config.ACTIVE_LABELS), "มีคำซ้ำใน ACTIVE_LABELS"
        assert 0 < config.CONFIDENCE_THRESHOLD < 1
        assert config.STABLE_FRAMES_REQUIRED >= 1
        assert 1 <= config.MAX_HANDS <= 2
        return f"{len(config.ACTIVE_LABELS)} คำ: {', '.join(config.ACTIVE_LABELS[:4])} ..."

    print()
    for line in PASSED:
        print(f"  [ผ่าน]   {line}")
    for name, detail in FAILED:
        print(f"  [ไม่ผ่าน] {name}")
        print("           " + detail.replace("\n", "\n           "))

    print()
    print("=" * 62)
    print(f"ผลรวม: ผ่าน {len(PASSED)} ข้อ, ไม่ผ่าน {len(FAILED)} ข้อ")
    print("=" * 62)

    if FAILED:
        return 1

    print("\nสภาพแวดล้อมพร้อมใช้งาน ขั้นตอนถัดไป:")
    print("  1. python -m src.collect     เก็บข้อมูลท่ามือจากกล้อง")
    print("  2. python -m src.train       เทรนโมเดล")
    print("  3. python -m src.translate   แปลภาษามือเป็นข้อความ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
