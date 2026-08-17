"""รับส่งชุดข้อมูลระหว่างคอมพิวเตอร์กับแอปมือถือ

รัน:
    run transfer export              แปลง data/landmarks.csv -> data/signdata.json
    run transfer export --out X.json ระบุชื่อไฟล์ปลายทางเอง
    run transfer import X.json       เพิ่มข้อมูลจากไฟล์ JSON เข้า data/landmarks.csv
    run transfer import X.json --replace   แทนที่ข้อมูลเดิมทั้งหมด

ทำไมแชร์กันได้: ทั้งสองฝั่งใช้ feature vector 128 ค่าที่คำนวณด้วยวิธีเดียวกัน
(ตรวจสอบได้ด้วย `run parity`) สิ่งที่แชร์คือ **ชุดข้อมูล** ไม่ใช่ตัวโมเดล
เพราะสองฝั่งใช้ไลบรารีคนละตัว แต่ละฝั่งจึงเทรนโมเดลของตัวเองจากข้อมูลชุดเดียวกัน
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

from .console import enable_utf8_output  # noqa: E402
from .dataset import append_sample, count_samples, load_dataset  # noqa: E402
from .features import FEATURE_DIMS  # noqa: E402

enable_utf8_output()

DATASET_FORMAT = "sign-language-translator-dataset"
DATASET_VERSION = 1


def do_export(output: Path) -> int:
    try:
        X, y = load_dataset(config.DATASET_CSV)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n{exc}")
        return 1

    payload = {
        "format": DATASET_FORMAT,
        "version": DATASET_VERSION,
        "featureDims": FEATURE_DIMS,
        "exportedAt": datetime.now(timezone.utc).isoformat(),
        "source": "desktop",
        "sampleCount": len(X),
        "samples": [
            {"label": str(label), "vector": [round(float(value), 6) for value in row]}
            for label, row in zip(y, X)
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    counts = Counter(str(label) for label in y)
    size_mb = output.stat().st_size / 1e6
    print(f"ส่งออก {len(X)} ตัวอย่าง จาก {len(counts)} คำ -> {output} ({size_mb:.1f} MB)")
    for label, count in sorted(counts.items()):
        print(f"    {label:<16} {count}")
    print("\nนำไฟล์นี้ไปเปิดในแอปมือถือ แล้วกด 'นำเข้าข้อมูล' ที่แท็บเทรน")
    return 0


def do_import(source: Path, replace: bool) -> int:
    if not source.exists():
        print(f"ไม่พบไฟล์ {source}")
        return 1

    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ไฟล์นี้ไม่ใช่ JSON ที่ถูกต้อง: {exc}")
        return 1

    if payload.get("format") != DATASET_FORMAT:
        print("ไม่ใช่ไฟล์ชุดข้อมูลของโปรเจคนี้")
        return 1

    dims = payload.get("featureDims")
    if dims != FEATURE_DIMS:
        print(f"จำนวนมิติไม่ตรงกัน (ไฟล์มี {dims} ต้องการ {FEATURE_DIMS})")
        print("อาจเกิดจากแอปมือถือกับคอมพิวเตอร์เป็นคนละรุ่น ลองรัน `run parity` ตรวจสอบ")
        return 1

    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        print("ไฟล์นี้ไม่มีข้อมูลตัวอย่างอยู่เลย")
        return 1

    # ตรวจทุกแถวให้ครบก่อนเขียนจริง จะได้ไม่เขียนข้อมูลเสียลงไปครึ่ง ๆ กลาง ๆ
    prepared: list[tuple[str, np.ndarray]] = []
    for index, sample in enumerate(samples, start=1):
        label = sample.get("label")
        vector = sample.get("vector")
        if not isinstance(label, str) or not label:
            print(f"ตัวอย่างลำดับที่ {index} ไม่มีชื่อคำกำกับ")
            return 1
        if not isinstance(vector, list) or len(vector) != FEATURE_DIMS:
            print(f"ตัวอย่างลำดับที่ {index} มีจำนวนค่าไม่ครบ {FEATURE_DIMS}")
            return 1
        prepared.append((label, np.asarray(vector, dtype=np.float32)))

    before = sum(count_samples(config.DATASET_CSV).values())

    if replace and config.DATASET_CSV.exists():
        backup = config.DATASET_CSV.with_suffix(".csv.bak")
        config.DATASET_CSV.replace(backup)
        print(f"สำรองข้อมูลเดิม {before} ตัวอย่างไว้ที่ {backup}")

    for label, vector in prepared:
        append_sample(config.DATASET_CSV, label, vector)

    counts = count_samples(config.DATASET_CSV)
    total = sum(counts.values())
    origin = payload.get("source", "ไม่ทราบ")
    print(f"\nนำเข้า {len(prepared)} ตัวอย่าง (จาก {origin}) เรียบร้อย")
    print(f"ชุดข้อมูลตอนนี้: {total} ตัวอย่าง จาก {len(counts)} คำ")
    for label, count in sorted(counts.items()):
        print(f"    {label:<16} {count}")
    print("\nขั้นตอนถัดไป: run train")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="รับส่งชุดข้อมูลกับแอปมือถือ")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="ส่งออกเป็นไฟล์ JSON")
    export_parser.add_argument(
        "--out", type=Path, default=config.DATA_DIR / "signdata.json", help="ไฟล์ปลายทาง"
    )

    import_parser = subparsers.add_parser("import", help="นำเข้าจากไฟล์ JSON")
    import_parser.add_argument("source", type=Path, help="ไฟล์ JSON ที่ส่งออกจากแอปมือถือ")
    import_parser.add_argument(
        "--replace", action="store_true", help="แทนที่ข้อมูลเดิมทั้งหมด (สำรองไฟล์เดิมให้อัตโนมัติ)"
    )

    args = parser.parse_args()

    if args.command == "export":
        return do_export(args.out)
    return do_import(args.source, args.replace)


if __name__ == "__main__":
    raise SystemExit(main())
