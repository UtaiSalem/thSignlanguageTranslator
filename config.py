"""ค่าตั้งต้นทั้งหมดของโปรเจค — แก้ไฟล์นี้ไฟล์เดียวก็เปลี่ยนพฤติกรรมทั้งระบบได้"""

from pathlib import Path

# ---------------------------------------------------------------- โครงสร้างไฟล์
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

DATASET_CSV = DATA_DIR / "landmarks.csv"       # ชุดข้อมูลที่เก็บจากกล้อง
CLASSIFIER_PATH = MODELS_DIR / "classifier.joblib"
REPORT_PATH = DOCS_DIR / "training_report.txt"

# โมเดลตรวจจับมือของ MediaPipe (ดาวน์โหลดอัตโนมัติถ้ายังไม่มี ดู src/hand_tracker.py)
HAND_LANDMARKER_TASK = MODELS_DIR / "hand_landmarker.task"
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# ---------------------------------------------------------------- กล้อง
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MIRROR_PREVIEW = True   # พลิกภาพซ้าย-ขวาให้เหมือนส่องกระจก (ใช้งานง่ายกว่า)

# ---------------------------------------------------------------- การตรวจจับมือ
MAX_HANDS = 2           # ภาษามือไทยหลายคำใช้สองมือ จึงเผื่อไว้ 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# ---------------------------------------------------------------- ชุดคำที่จะสอน
# ภาษามือไทย: พยัญชนะ ก-ฮ (ท่ามือค้างนิ่ง เหมาะกับโมเดลแบบภาพนิ่งที่สุด)
LABELS_TH_ALPHABET = [
    "ก", "ข", "ค", "ง", "จ", "ฉ", "ช", "ซ", "ญ", "ด",
    "ต", "ถ", "ท", "น", "บ", "ป", "ผ", "ฝ", "พ", "ฟ",
    "ม", "ย", "ร", "ล", "ว", "ส", "ห", "อ", "ฮ",
]

# คำพื้นฐานที่ใช้บ่อยในชีวิตประจำวัน
LABELS_TH_WORDS = [
    "สวัสดี",
    "ขอบคุณ",
    "ขอโทษ",
    "ใช่",
    "ไม่",
    "รัก",
    "ช่วยด้วย",
    "เข้าใจ",
    "ไม่เข้าใจ",
    "ชื่ออะไร",
]

# ชุดที่ใช้งานจริงตอนนี้ — สลับได้ระหว่าง LABELS_TH_WORDS / LABELS_TH_ALPHABET
# หรือเขียนลิสต์ของคุณเองก็ได้
ACTIVE_LABELS = LABELS_TH_WORDS

# ---------------------------------------------------------------- การเก็บข้อมูล
SAMPLES_PER_CLASS_TARGET = 200   # จำนวนตัวอย่างที่แนะนำต่อ 1 คำ
BURST_INTERVAL_SEC = 0.08        # ระยะห่างระหว่างเฟรมตอนกดเก็บแบบต่อเนื่อง

# ---------------------------------------------------------------- การเทรน
TEST_SIZE = 0.2
RANDOM_STATE = 42
MLP_HIDDEN_LAYERS = (256, 128)
MLP_MAX_ITER = 800

# ---------------------------------------------------------------- การแปลผล
CONFIDENCE_THRESHOLD = 0.75   # ต่ำกว่านี้ถือว่า "ไม่มั่นใจ" ไม่แสดงผล
STABLE_FRAMES_REQUIRED = 8    # ต้องทายคำเดิมติดกันกี่เฟรมถึงจะยอมรับ
COOLDOWN_SEC = 1.0            # หน่วงเวลาก่อนรับคำถัดไป กันคำซ้ำรัว ๆ

# ---------------------------------------------------------------- การแสดงผลภาษาไทย
# cv2.putText วาดภาษาไทยไม่ได้ จึงต้องใช้ Pillow + ฟอนต์ไทย (ดู src/thai_text.py)
THAI_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\leelawui.ttf",   # Leelawadee UI
    r"C:\Windows\Fonts\leelawad.ttf",   # Leelawadee
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\browa.ttf",      # Browallia New
    "/usr/share/fonts/truetype/tlwg/Garuda.ttf",
    "/System/Library/Fonts/Supplemental/Thonburi.ttc",
]
