"""ตรวจสอบว่าโค้ดฝั่ง Python กับฝั่ง JavaScript คำนวณ feature ได้ค่าตรงกัน

รัน:  run parity

ทำไมต้องมี: `src/features.py` (คอมพิวเตอร์) กับ `app/js/features.js` (มือถือ)
เป็นโค้ดคนละภาษาที่ต้องให้ผลเหมือนกันทุกหลัก ถ้าไม่ตรงกัน ข้อมูลที่เก็บจากมือถือ
จะใช้เทรนบนคอมพิวเตอร์ไม่ได้ (และกลับกัน) โดยที่ไม่มีอะไรฟ้อง error เลย
โมเดลจะแค่แม่นยำต่ำลงอย่างอธิบายไม่ถูก ซึ่งหาสาเหตุยากมาก

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
    build_feature_vector,
    mirror_feature_vector,
)

enable_utf8_output()

ROOT = Path(__file__).resolve().parent
TOLERANCE = 1e-6


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
            }
        )
    return results


def main() -> int:
    print("=" * 66)
    print("ตรวจสอบความตรงกันระหว่าง Python (คอมพิวเตอร์) กับ JavaScript (มือถือ)")
    print("=" * 66)

    if shutil.which("node") is None:
        print("\nไม่พบ Node.js ในเครื่อง — ข้ามการตรวจสอบนี้")
        print("ติดตั้งได้จาก https://nodejs.org แล้วรันใหม่")
        return 1

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

    print(f"ตรวจสอบ {len(cases)} กรณี (ยอมรับความต่างได้ไม่เกิน {TOLERANCE})\n")

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

    print()
    print("=" * 66)
    if failures:
        print(f"ไม่ตรงกัน {failures} กรณี — src/features.py กับ app/js/features.js ไม่สอดคล้องกัน")
        print("ข้อมูลจากมือถือกับคอมพิวเตอร์จะใช้ร่วมกันไม่ได้จนกว่าจะแก้ให้ตรง")
        return 1

    print("ตรงกันทุกกรณี — ข้อมูลจากมือถือกับคอมพิวเตอร์ใช้ร่วมกันได้")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
