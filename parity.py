"""ตรวจสอบว่าฝั่งคอมพิวเตอร์ (Python) กับฝั่งมือถือ (JavaScript) ให้เวกเตอร์ตรงกัน

รัน:  run parity

ทำไมต้องมี: `src/features.py` (คอมพิวเตอร์) กับ `app/js/features.js` (มือถือ)
เป็นโค้ดคนละภาษาที่ต้องให้ผลเหมือนกันทุกหลัก ถ้าไม่ตรงกัน ข้อมูลที่เก็บจากมือถือ
จะใช้เทรนบนคอมพิวเตอร์ไม่ได้ (และกลับกัน) โดยที่ไม่มีอะไรฟ้อง error เลย
โมเดลจะแค่แม่นยำต่ำลงอย่างอธิบายไม่ถูก ซึ่งหาสาเหตุยากมาก

ตรวจสองขั้น:
    1. ฟังก์ชันคำนวณ feature ให้ค่าตรงกันไหม (ป้อน landmark ชุดเดียวกันเข้าไป)
    2. ทั้ง **เส้นทาง** ให้ค่าตรงกันไหม — ฟังก์ชันตรงกันแล้วยังไม่พอ เพราะสองฝั่ง
       ป้อนภาพเข้า MediaPipe คนละแบบ (คอมพิวเตอร์พลิกกระจกก่อนตรวจจับ มือถือไม่พลิก)
       ทำให้ชื่อมือซ้าย-ขวาและแกน x กลับด้านกัน ขั้นนี้จับความไม่ตรงกันแบบนั้น

**แก้ไฟล์ features ไฟล์ใดไฟล์หนึ่งเมื่อไหร่ ให้รันคำสั่งนี้ทุกครั้ง**
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.console import enable_utf8_output  # noqa: E402
from src.features import (  # noqa: E402
    FEATURE_DIMS,
    build_canonical_feature_vector,
    build_feature_vector,
    mirror_feature_vector,
)

enable_utf8_output()

ROOT = Path(__file__).resolve().parent
TOLERANCE = 1e-6

# การตรวจสอบ pipeline ต้องยอมความต่างได้มากกว่า เพราะฝั่งคอมพิวเตอร์คำนวณ 1.0 - x
# ด้วย float32 เพิ่มอีกขั้นหนึ่ง จึงมีค่าปัดเศษสะสมที่ไม่ได้เกิดกับฝั่งมือถือ
PIPELINE_TOLERANCE = 1e-5


class FakeHand:
    def __init__(self, landmarks, handedness):
        self.landmarks = landmarks
        self.handedness = handedness


def make_cases() -> list[dict]:
    """สร้างกรณีทดสอบที่ครอบคลุมทุกสถานการณ์ที่เจอได้จริง"""
    rng = np.random.default_rng(20260817)

    def hand(seed_offset: float, handedness: str) -> dict:
        points = rng.random((21, 3)).astype(np.float64) * 0.4 + seed_offset
        return {"handedness": handedness, "landmarks": points.tolist()}

    cases = [
        {"name": "ไม่มีมือ", "hands": []},
        {"name": "มือขวาข้างเดียว", "hands": [hand(0.30, "Right")]},
        {"name": "มือซ้ายข้างเดียว", "hands": [hand(0.25, "Left")]},
        {"name": "สองมือ", "hands": [hand(0.20, "Left"), hand(0.55, "Right")]},
        {"name": "มืออยู่มุมจอ", "hands": [hand(0.02, "Right")]},
        {"name": "มือใกล้กล้องมาก", "hands": [hand(0.10, "Right")]},
    ]

    # กรณีสุดขั้ว: ทุกจุดซ้อนกันหมด ต้องไม่เกิดการหารด้วยศูนย์ทั้งสองฝั่ง
    cases.append(
        {
            "name": "จุดซ้อนกันทั้งหมด (กันหารศูนย์)",
            "hands": [{"handedness": "Right", "landmarks": [[0.5, 0.5, 0.0]] * 21}],
        }
    )
    # กรณีที่ค่า z ติดลบ ซึ่งเกิดขึ้นจริงเมื่อมือเอียงเข้าหากล้อง
    negative = (rng.random((21, 3)) * 0.4 + 0.3)
    negative[:, 2] -= 0.5
    cases.append(
        {"name": "ค่า z ติดลบ", "hands": [{"handedness": "Left", "landmarks": negative.tolist()}]}
    )

    # กรณีที่เจาะจงทดสอบท่อนคู่มือ: รูปมือทั้งสองข้างเหมือนกันทุกกรณีข้างล่าง
    # ต่างกันแค่ระยะห่างและขนาด ก่อนมีท่อนคู่มือ ทุกกรณีนี้ให้เวกเตอร์เดียวกันเป๊ะ
    def pair(separation: float, right_scale: float) -> dict:
        left = rng.random((21, 3)) * 0.25 + np.array([0.10, 0.40, 0.0])
        shape = rng.random((21, 3)) * 0.25
        right = shape * right_scale + np.array([0.10 + separation, 0.40, 0.0])
        return {
            "hands": [
                {"handedness": "Left", "landmarks": left.tolist()},
                {"handedness": "Right", "landmarks": right.tolist()},
            ]
        }

    cases += [
        {"name": "สองมือประกบกัน", **pair(0.04, 1.0)},
        {"name": "สองมือแยกห่างกัน", **pair(0.45, 1.0)},
        {"name": "สองมือขนาดต่างกันมาก", **pair(0.25, 2.2)},
        {"name": "สองมือวางซ้อนแนวตั้ง", "hands": [
            {"handedness": "Left", "landmarks": (rng.random((21, 3)) * 0.2 + np.array([0.4, 0.15, 0.0])).tolist()},
            {"handedness": "Right", "landmarks": (rng.random((21, 3)) * 0.2 + np.array([0.4, 0.65, 0.0])).tolist()},
        ]},
    ]

    return cases


def python_results(cases: list[dict]) -> list[dict]:
    results = []
    for case in cases:
        hands = [
            FakeHand(np.array(h["landmarks"], dtype=np.float32), h["handedness"])
            for h in case["hands"]
        ]
        vector = build_feature_vector(hands)
        results.append(
            {
                "name": case["name"],
                "vector": vector.tolist(),
                "mirrored": mirror_feature_vector(vector).tolist(),
                # เส้นทางจริงของฝั่งคอมพิวเตอร์ ซึ่งเห็นภาพที่พลิกกระจกแล้ว
                "desktopPipeline": desktop_pipeline_vector(case["hands"]).tolist(),
            }
        )
    return results


def desktop_pipeline_vector(case_hands: list[dict]) -> np.ndarray:
    """จำลองเส้นทางของฝั่งคอมพิวเตอร์ทั้งเส้น แล้วคืนเวกเตอร์ในคอนเวนชันกลาง

    `src/collect.py` พลิกเฟรมด้วย cv2.flip ก่อนส่งเข้า MediaPipe ผลที่ MediaPipe
    คืนมาจึงต่างจากฝั่งมือถือสองอย่าง: พิกัด x กลับด้าน (x -> 1-x) และชื่อมือสลับข้าง
    จำลองสองอย่างนั้นตรงนี้ แล้วส่งเข้า build_canonical_feature_vector
    ผลลัพธ์ต้องเท่ากับเวกเตอร์ที่ฝั่งมือถือได้จากภาพดิบเป๊ะ ๆ
    """
    swap = {"Left": "Right", "Right": "Left"}
    hands = []
    for hand in case_hands:
        points = np.array(hand["landmarks"], dtype=np.float32)
        flipped = points.copy()
        flipped[:, 0] = np.float32(1.0) - flipped[:, 0]
        hands.append(FakeHand(flipped, swap.get(hand["handedness"], hand["handedness"])))
    return build_canonical_feature_vector(hands, source_is_mirrored=True)


def main() -> int:
    print("=" * 66)
    print("ตรวจสอบความตรงกันระหว่าง Python (คอมพิวเตอร์) กับ JavaScript (มือถือ)")
    print("=" * 66)

    if shutil.which("node") is None:
        # คืนค่า 0 เพราะ "ไม่มีเครื่องมือให้ตรวจ" ไม่ใช่ "ตรวจแล้วไม่ผ่าน"
        # แต่ต้องพูดให้ชัดว่ายังไม่ได้ตรวจอะไรเลย ไม่ใช่ผ่าน
        print("\n" + "=" * 66)
        print("ยังไม่ได้ตรวจสอบ — ไม่พบ Node.js ในเครื่อง")
        print("=" * 66)
        print("การตรวจสอบนี้ต้องรันโค้ดฝั่ง JavaScript จึงต้องมี Node.js")
        print("ติดตั้งจาก https://nodejs.org แล้วรัน `run parity` อีกครั้ง")
        print("\nถ้าคุณแก้ features.py หรือ features.js ไป อย่าถือว่าผ่าน")
        print("จนกว่าจะได้รันคำสั่งนี้จริง")
        return 0

    cases = make_cases()
    workdir = Path(tempfile.mkdtemp())
    try:
        fixtures_path = workdir / "fixtures.json"
        results_path = workdir / "results.json"
        fixtures_path.write_text(
            json.dumps({"cases": cases}, ensure_ascii=False), encoding="utf-8"
        )

        process = subprocess.run(
            ["node", str(ROOT / "tools" / "parity_check.mjs"), str(fixtures_path), str(results_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if process.returncode != 0:
            print("\nรันฝั่ง JavaScript ไม่สำเร็จ:")
            print(process.stdout)
            print(process.stderr)
            return 1

        js_payload = json.loads(results_path.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    js_results = js_payload["results"]
    py_results = python_results(cases)

    print(f"\nจำนวนมิติของ feature: Python={FEATURE_DIMS}, JavaScript={js_payload['featureDims']}")
    if FEATURE_DIMS != js_payload["featureDims"]:
        print("จำนวนมิติไม่ตรงกัน — หยุดตรวจสอบ")
        return 1

    print("\n[1/2] ฟังก์ชันคำนวณ feature ให้ค่าตรงกันไหม")
    print(f"      ตรวจสอบ {len(cases)} กรณี (ยอมรับความต่างได้ไม่เกิน {TOLERANCE})\n")

    failures = 0
    for py, js in zip(py_results, js_results):
        vector_diff = np.abs(np.array(py["vector"]) - np.array(js["vector"])).max()
        mirror_diff = np.abs(np.array(py["mirrored"]) - np.array(js["mirrored"])).max()
        worst = max(vector_diff, mirror_diff)
        ok = worst <= TOLERANCE
        if not ok:
            failures += 1
        mark = "ผ่าน  " if ok else "ไม่ผ่าน"
        print(f"  [{mark}] {py['name']:<32} ต่างกันมากที่สุด {worst:.3e}")

    # ขั้นที่สองสำคัญไม่แพ้กัน: ฟังก์ชันตรงกันแล้วยังไม่พอ ถ้าสองฝั่งป้อน "ภาพ"
    # เข้าไปคนละแบบ (ฝั่งคอมพิวเตอร์พลิกกระจกก่อนตรวจจับ ฝั่งมือถือไม่พลิก)
    # เวกเตอร์ที่ได้ก็ยังลงช่องมือสลับข้างกันอยู่ดี
    print("\n[2/2] ทั้งสองเส้นทางให้เวกเตอร์เดียวกันจากท่ามือเดียวกันไหม")
    print(f"      (คอมพิวเตอร์เห็นภาพพลิกกระจก มือถือเห็นภาพดิบ, ยอมได้ไม่เกิน {PIPELINE_TOLERANCE})\n")

    for py, js in zip(py_results, js_results):
        diff = np.abs(np.array(py["desktopPipeline"]) - np.array(js["vector"])).max()
        ok = diff <= PIPELINE_TOLERANCE
        if not ok:
            failures += 1
        mark = "ผ่าน  " if ok else "ไม่ผ่าน"
        print(f"  [{mark}] {py['name']:<32} ต่างกันมากที่สุด {diff:.3e}")

    print()
    print("=" * 66)
    if failures:
        print(f"ไม่ตรงกัน {failures} กรณี — ฝั่งคอมพิวเตอร์กับฝั่งมือถือไม่สอดคล้องกัน")
        print("ข้อมูลจากมือถือกับคอมพิวเตอร์จะใช้ร่วมกันไม่ได้จนกว่าจะแก้ให้ตรง")
        print("ถ้าไม่ผ่านเฉพาะขั้นที่ 2 ให้ดูการแปลงคอนเวนชันใน src/features.py")
        return 1

    print("ตรงกันทุกกรณี — ข้อมูลจากมือถือกับคอมพิวเตอร์ใช้ร่วมกันได้")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
