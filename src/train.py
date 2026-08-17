"""ขั้นตอนที่ 2 — เทรนโมเดลจำแนกท่ามือ

รัน:  python -m src.train
      python -m src.train --model rf        (ลองใช้ Random Forest เทียบดู)
      python -m src.train --no-augment      (ปิดการเพิ่มข้อมูลแบบกลับด้าน)

โมเดลที่ใช้เป็นโครงข่ายประสาทเทียมขนาดเล็ก (MLP) รับ input 128 ค่าจาก
src/features.py ซึ่งเป็น "รูปทรงมือ" ที่ตัดผลของตำแหน่งและขนาดออกไปแล้ว
เพราะ input ถูกกลั่นมาดีตั้งแต่ต้น โมเดลจึงเล็กมาก เทรนเสร็จในไม่กี่วินาที
และใช้ข้อมูลเพียงหลักร้อยตัวอย่างต่อคำ — ต่างจากการเทรน CNN จากรูปภาพดิบ
ที่ต้องใช้ข้อมูลหลายพันรูปและใช้เวลานานกว่ามาก
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

from .dataset import load_dataset  # noqa: E402
from .features import mirror_feature_vector  # noqa: E402

MIN_SAMPLES_PER_CLASS = 10


def augment_with_mirror(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """เพิ่มข้อมูลเท่าตัวด้วยการกลับด้านซ้าย-ขวา (ดูคำอธิบายใน features.py)"""
    mirrored = np.array([mirror_feature_vector(row) for row in X], dtype=np.float32)
    return np.vstack([X, mirrored]), np.concatenate([y, y])


def build_model(kind: str) -> Pipeline:
    """สร้าง pipeline: ปรับสเกล -> จำแนกประเภท

    StandardScaler ทำให้ทุก feature มีค่าเฉลี่ย 0 และส่วนเบี่ยงเบน 1
    ซึ่งช่วยให้โครงข่ายประสาทเทียมลู่เข้าเร็วขึ้นมาก
    """
    if kind == "rf":
        classifier = RandomForestClassifier(
            n_estimators=400,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        classifier = MLPClassifier(
            hidden_layer_sizes=config.MLP_HIDDEN_LAYERS,
            max_iter=config.MLP_MAX_ITER,
            random_state=config.RANDOM_STATE,
            early_stopping=True,
            n_iter_no_change=25,
        )

    return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])


def check_dataset(y: np.ndarray) -> list[str]:
    """ตรวจคุณภาพชุดข้อมูลก่อนเทรน แล้วคืนรายการคำเตือน"""
    warnings: list[str] = []
    counts = Counter(y)

    for label, count in sorted(counts.items(), key=lambda item: item[1]):
        if count < MIN_SAMPLES_PER_CLASS:
            warnings.append(
                f"คำว่า '{label}' มีแค่ {count} ตัวอย่าง "
                f"(ควรมีอย่างน้อย {MIN_SAMPLES_PER_CLASS} ยิ่งมากยิ่งดี)"
            )

    fewest, most = min(counts.values()), max(counts.values())
    if most > fewest * 3:
        warnings.append(
            f"จำนวนตัวอย่างแต่ละคำต่างกันมาก ({fewest} ถึง {most}) "
            f"โมเดลจะเอนเอียงไปทางคำที่มีข้อมูลเยอะ ควรเก็บเพิ่มให้ใกล้เคียงกัน"
        )

    return warnings


def format_report(
    kind: str,
    counts: Counter,
    train_size: int,
    test_size: int,
    accuracy: float,
    report_text: str,
    matrix: np.ndarray,
    labels: list[str],
) -> str:
    """จัดรายงานผลการเทรนเป็นข้อความ เก็บไว้อ้างอิงในเอกสารโปรเจค"""
    lines = [
        "=" * 70,
        "รายงานผลการเทรนโมเดลแปลภาษามือไทย",
        "=" * 70,
        "",
        f"ชนิดโมเดล        : {kind}",
        f"จำนวนคำ          : {len(labels)}",
        f"ตัวอย่างสำหรับเทรน : {train_size}",
        f"ตัวอย่างสำหรับทดสอบ: {test_size}",
        f"ความแม่นยำรวม     : {accuracy * 100:.2f}%",
        "",
        "จำนวนตัวอย่างต่อคำ (ข้อมูลดิบก่อนเพิ่มข้อมูล)",
        "-" * 70,
    ]
    lines += [f"  {label:<14} {counts[label]}" for label in labels]
    lines += ["", "ผลแยกรายคำ", "-" * 70, report_text, "", "Confusion matrix", "-" * 70]

    width = max(max((len(label) for label in labels), default=4), 4) + 2
    lines.append(" " * width + "".join(f"{label:>{width}}" for label in labels))
    for label, row in zip(labels, matrix):
        lines.append(f"{label:<{width}}" + "".join(f"{value:>{width}}" for value in row))

    lines += [
        "",
        "วิธีอ่าน confusion matrix: แถว = คำที่ทำจริง, คอลัมน์ = คำที่โมเดลทาย",
        "ตัวเลขนอกแนวทแยงมุมคือความสับสน ถ้าคู่ไหนสับสนกันบ่อย แปลว่าท่ามือสองท่านั้น",
        "คล้ายกันเกินไป ให้เก็บข้อมูลของสองคำนั้นเพิ่ม หรือเลือกใช้ท่าที่ต่างกันชัดกว่า",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="เทรนโมเดลจำแนกท่ามือ")
    parser.add_argument("--model", choices=["mlp", "rf"], default="mlp", help="ชนิดโมเดล")
    parser.add_argument("--no-augment", action="store_true", help="ไม่เพิ่มข้อมูลแบบกลับด้าน")
    args = parser.parse_args()

    print(f"โหลดข้อมูลจาก {config.DATASET_CSV}")
    try:
        X, y = load_dataset(config.DATASET_CSV)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n{exc}")
        return 1
    raw_counts = Counter(y)
    labels = sorted(raw_counts)
    print(f"พบ {len(X)} ตัวอย่าง จาก {len(labels)} คำ\n")

    for warning in check_dataset(y):
        print(f"  คำเตือน: {warning}")

    if len(labels) < 2:
        print("\nต้องมีอย่างน้อย 2 คำถึงจะเทรนได้ — กลับไปเก็บข้อมูลเพิ่มด้วย python -m src.collect")
        return 1

    # แบ่งข้อมูลก่อนเพิ่มข้อมูลเสมอ ไม่งั้นภาพกลับด้านของตัวอย่างเดียวกัน
    # จะไปโผล่ทั้งในชุดเทรนและชุดทดสอบ ทำให้คะแนนสูงเกินจริง (data leakage)
    stratify = y if min(raw_counts.values()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=stratify
    )

    if not args.no_augment:
        before = len(X_train)
        X_train, y_train = augment_with_mirror(X_train, y_train)
        print(f"\nเพิ่มข้อมูลแบบกลับด้าน: {before} -> {len(X_train)} ตัวอย่าง")

    print(f"\nกำลังเทรนโมเดล ({args.model}) ...")
    model = build_model(args.model)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"ความแม่นยำบนชุดทดสอบ: {accuracy * 100:.2f}%\n")

    test_labels = sorted(set(y_test) | set(y_pred))
    report_text = classification_report(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred, labels=test_labels)
    print(report_text)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "labels": list(model.classes_),
            "model_kind": args.model,
            "accuracy": float(accuracy),
            "augmented": not args.no_augment,
        },
        config.CLASSIFIER_PATH,
    )
    print(f"บันทึกโมเดลแล้ว -> {config.CLASSIFIER_PATH}")

    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_PATH.write_text(
        format_report(
            args.model, raw_counts, len(X_train), len(X_test),
            accuracy, report_text, matrix, test_labels,
        ),
        encoding="utf-8",
    )
    print(f"บันทึกรายงานแล้ว -> {config.REPORT_PATH}")
    print("\nขั้นตอนถัดไป: python -m src.translate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
