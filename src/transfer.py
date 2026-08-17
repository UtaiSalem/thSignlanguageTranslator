"""รับส่งชุดข้อมูลระหว่างคอมพิวเตอร์กับแอปมือถือ

รัน:
    run transfer export              แปลง data/landmarks.csv -> data/signdata.json
    run transfer export --out X.json ระบุชื่อไฟล์ปลายทางเอง
    run transfer import X.json       เพิ่มข้อมูลจากไฟล์ JSON เข้า data/landmarks.csv
    run transfer import X.json --replace   แทนที่ข้อมูลเดิมทั้งหมด
    run transfer convert-legacy      ขยายชุดข้อมูลเก่าให้ครบจำนวนมิติปัจจุบัน
    run transfer convert-legacy --mirror   สลับคอนเวนชันมือซ้าย-ขวาด้วย

ทำไมแชร์กันได้: ทั้งสองฝั่งใช้ feature vector 128 ค่าที่คำนวณด้วยวิธีเดียวกัน
(ตรวจสอบได้ด้วย `run parity`) สิ่งที่แชร์คือ **ชุดข้อมูล** ไม่ใช่ตัวโมเดล
เพราะสองฝั่งใช้ไลบรารีคนละตัว แต่ละฝั่งจึงเทรนโมเดลของตัวเองจากข้อมูลชุดเดียวกัน
"""

from __future__ import annotations

import argparse
import csv
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
from .features import (  # noqa: E402
    FEATURE_DIMS,
    LEGACY_FEATURE_DIMS,
    mirror_feature_vector,
    upgrade_legacy_vector,
)

enable_utf8_output()

DATASET_FORMAT = "sign-language-translator-dataset"
DATASET_VERSION = 3

# ไฟล์รุ่นเก่ายังนำเข้าได้ แต่ต้องแปลงก่อน มีสองเรื่องที่ต่างกัน และเป็นอิสระจากกัน
#
#   รุ่น 1 — 128 มิติ และฝั่งคอมพิวเตอร์ยังเก็บในคอนเวนชันภาพกระจก
#            (x กลับด้าน ช่องมือสลับข้าง) ส่วนฝั่งมือถือเป็นคอนเวนชันกลางอยู่แล้ว
#   รุ่น 2 — 128 มิติ คอนเวนชันมือตรงกันทั้งสองฝั่งแล้ว แต่ยังไม่มีท่อนคู่มือ
#   รุ่น 3 — 133 มิติ มีท่อนคู่มือ (ดู src/features.py)
LEGACY_VERSIONS = (1, 2)
MIRRORED_CONVENTION_VERSION = 1


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

    version = payload.get("version")
    origin = payload.get("source", "ไม่ทราบ")
    if version not in (*LEGACY_VERSIONS, DATASET_VERSION):
        print(f"ไฟล์นี้เป็นรุ่น {version} ซึ่งโปรแกรมรุ่นนี้อ่านไม่ได้ (อ่านได้ถึงรุ่น {DATASET_VERSION})")
        return 1

    needs_upgrade = version in LEGACY_VERSIONS
    expected_dims = LEGACY_FEATURE_DIMS if needs_upgrade else FEATURE_DIMS

    dims = payload.get("featureDims")
    if dims != expected_dims:
        print(f"จำนวนมิติไม่ตรงกัน (ไฟล์รุ่น {version} มี {dims} ควรเป็น {expected_dims})")
        print("อาจเกิดจากแอปมือถือกับคอมพิวเตอร์เป็นคนละรุ่น ลองรัน `run parity` ตรวจสอบ")
        return 1

    # ไฟล์รุ่น 1 จากคอมพิวเตอร์อยู่ในคอนเวนชันภาพกระจก ต้องแปลงเป็นคอนเวนชันกลาง
    # ก่อนเอามาปนกับข้อมูลใหม่ ไม่งั้นชุดข้อมูลจะมีทั้งสองคอนเวนชันคลุกกันอยู่
    needs_mirror = version == MIRRORED_CONVENTION_VERSION and origin == "desktop"

    if needs_upgrade:
        print(f"ไฟล์นี้เป็นรุ่น {version} — แปลงให้อัตโนมัติ")
        print(f"  ขยายจาก {LEGACY_FEATURE_DIMS} เป็น {FEATURE_DIMS} มิติ (เติมท่อนคู่มือ)")
        print("    ท่อนคู่มือของข้อมูลเก่ากู้กลับไม่ได้ จึงติดธงไว้ว่า 'ไม่รู้ค่า'")
        print("    ตัวอย่างที่ใช้มือข้างเดียวไม่เสียหายอะไร เพราะค่าที่ถูกคือศูนย์อยู่แล้ว")
        if needs_mirror:
            print("  สลับคอนเวนชันมือซ้าย-ขวาให้ตรงกับคอนเวนชันกลาง")
        elif version == MIRRORED_CONVENTION_VERSION:
            print(f"  ต้นทางคือ '{origin}' ซึ่งอยู่ในคอนเวนชันมือแบบใหม่อยู่แล้ว ไม่ต้องสลับ")

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
        if not isinstance(vector, list) or len(vector) != expected_dims:
            print(f"ตัวอย่างลำดับที่ {index} มีจำนวนค่าไม่ครบ {expected_dims}")
            return 1
        values = np.asarray(vector, dtype=np.float32)
        if needs_upgrade:
            values = upgrade_legacy_vector(values)
        if needs_mirror:
            values = mirror_feature_vector(values)
        prepared.append((label, values))

    before = sum(count_samples(config.DATASET_CSV).values())

    if replace and config.DATASET_CSV.exists():
        backup = config.DATASET_CSV.with_suffix(".csv.bak")
        config.DATASET_CSV.replace(backup)
        print(f"สำรองข้อมูลเดิม {before} ตัวอย่างไว้ที่ {backup}")

    for label, vector in prepared:
        append_sample(config.DATASET_CSV, label, vector)

    counts = count_samples(config.DATASET_CSV)
    total = sum(counts.values())
    print(f"\nนำเข้า {len(prepared)} ตัวอย่าง (จาก {origin}) เรียบร้อย")
    print(f"ชุดข้อมูลตอนนี้: {total} ตัวอย่าง จาก {len(counts)} คำ")
    for label, count in sorted(counts.items()):
        print(f"    {label:<16} {count}")
    print("\nขั้นตอนถัดไป: run train")
    return 0


def do_convert_legacy(apply_mirror: bool) -> int:
    """แปลง data/landmarks.csv ในเครื่องให้ทันสมัย

    มีสองงานที่ทำได้ และคุณสมบัติต่างกันมาก:

    1. **ขยายจำนวนมิติ** จาก 128 เป็น 133 (เติมท่อนคู่มือ) — ตรวจได้อัตโนมัติจาก
       หัวตาราง จึงทำให้เองเมื่อจำเป็น และไม่ทำซ้ำถ้าแปลงไปแล้ว
    2. **สลับคอนเวนชันมือซ้าย-ขวา** (`--mirror`) — ตรวจอัตโนมัติ **ไม่ได้** เพราะ
       ไฟล์ CSV ไม่ได้บันทึกไว้ว่าเก็บด้วยโค้ดรุ่นไหน ผู้ใช้ต้องสั่งเอง
       สั่งซ้ำสองครั้งจะได้ข้อมูลเดิมคืน (การกลับด้านสองครั้งหักล้างกันเอง)
    """
    if not config.DATASET_CSV.exists():
        print(f"ยังไม่มีไฟล์ข้อมูล {config.DATASET_CSV}")
        return 1

    with config.DATASET_CSV.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))

    if len(rows) < 2:
        print(f"ไฟล์ {config.DATASET_CSV} ยังไม่มีข้อมูล")
        return 1

    header, data_rows = rows[0], rows[1:]
    value_count = len(header) - 1

    if value_count == LEGACY_FEATURE_DIMS:
        needs_upgrade = True
    elif value_count == FEATURE_DIMS:
        needs_upgrade = False
    else:
        print(f"หัวตารางมี {value_count} คอลัมน์ค่า ซึ่งไม่ตรงกับรุ่นใดที่รู้จัก")
        print(f"(รุ่นเก่า {LEGACY_FEATURE_DIMS}, รุ่นปัจจุบัน {FEATURE_DIMS})")
        return 1

    if not needs_upgrade and not apply_mirror:
        print("ชุดข้อมูลนี้ทันสมัยอยู่แล้ว ไม่มีอะไรต้องแปลง")
        print("ถ้าต้องการสลับคอนเวนชันมือซ้าย-ขวา ให้ใส่ --mirror")
        return 0

    prepared: list[tuple[str, np.ndarray]] = []
    for index, row in enumerate(data_rows, start=1):
        if len(row) != len(header):
            print(f"แถวที่ {index} มีจำนวนคอลัมน์ไม่ตรงกับหัวตาราง — หยุดก่อนแก้ไฟล์")
            return 1
        values = np.asarray(row[1:], dtype=np.float32)
        if needs_upgrade:
            values = upgrade_legacy_vector(values)
        if apply_mirror:
            values = mirror_feature_vector(values)
        prepared.append((row[0], values))

    backup = config.DATASET_CSV.with_suffix(".csv.bak")
    config.DATASET_CSV.replace(backup)
    print(f"สำรองไฟล์เดิมไว้ที่ {backup}")

    for label, values in prepared:
        append_sample(config.DATASET_CSV, label, values)

    print(f"\nแปลง {len(prepared)} ตัวอย่างแล้ว -> {config.DATASET_CSV}")
    if needs_upgrade:
        print(f"  ขยายจาก {LEGACY_FEATURE_DIMS} เป็น {FEATURE_DIMS} มิติ (ท่อนคู่มือติดธง 'ไม่รู้ค่า')")
    if apply_mirror:
        print("  สลับคอนเวนชันมือซ้าย-ขวาแล้ว")
        print("  ถ้าความแม่นยำตกลงหลังแปลง แปลว่าข้อมูลชุดนี้อยู่ในคอนเวนชันใหม่อยู่แล้ว")
        print(f"  ให้สั่ง --mirror ซ้ำอีกครั้งเพื่อย้อนกลับ หรือกู้จาก {backup}")
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

    convert_parser = subparsers.add_parser(
        "convert-legacy",
        help="แปลงชุดข้อมูลรุ่นเก่าในเครื่องให้ทันสมัย",
    )
    convert_parser.add_argument(
        "--mirror",
        action="store_true",
        help="สลับคอนเวนชันมือซ้าย-ขวาด้วย (สำหรับข้อมูลที่เก็บก่อนรวมคอนเวนชัน)",
    )

    args = parser.parse_args()

    if args.command == "export":
        return do_export(args.out)
    if args.command == "convert-legacy":
        return do_convert_legacy(args.mirror)
    return do_import(args.source, args.replace)


if __name__ == "__main__":
    raise SystemExit(main())
