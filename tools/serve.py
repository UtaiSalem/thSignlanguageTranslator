"""เปิดเว็บเซิร์ฟเวอร์ในเครื่อง เพื่อทดสอบแอปมือถือบนคอมพิวเตอร์

รัน:  run serve

เปิดเบราว์เซอร์ไปที่ http://localhost:8000 แล้วแอปจะทำงานเหมือนบนมือถือทุกอย่าง
รวมถึงกล้องด้วย เพราะเบราว์เซอร์ถือว่า localhost เป็นแหล่งที่ปลอดภัย (secure context)

**ข้อควรรู้:** เปิดจากมือถือผ่านเลข IP ในวง LAN จะใช้กล้องไม่ได้
เพราะเบราว์เซอร์บังคับว่าต้องเป็น https เท่านั้นถ้าไม่ใช่ localhost
การทดสอบบนมือถือจริงให้ใช้ GitHub Pages (ดูหัวข้อการติดตั้งใน app/README.md)
"""

from __future__ import annotations

import argparse
import mimetypes
import socket
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "app"

sys.path.insert(0, str(ROOT))
from src.console import enable_utf8_output  # noqa: E402

enable_utf8_output()

# Python บางรุ่นยังไม่รู้จักนามสกุลเหล่านี้ ถ้าส่ง MIME type ผิด
# เบราว์เซอร์จะปฏิเสธไม่ยอมโหลด WASM และ ES module
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("application/manifest+json", ".webmanifest")


class AppRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # กันไม่ให้เบราว์เซอร์แคชระหว่างพัฒนา จะได้เห็นผลการแก้ไฟล์ทันที
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, format, *args):
        # แสดงเฉพาะข้อผิดพลาด ไม่ต้องรกด้วย log ของทุกไฟล์
        status = args[1] if len(args) > 1 else ""
        if str(status).startswith(("4", "5")):
            super().log_message(format, *args)


def local_ip() -> str:
    """หาเลข IP ของเครื่องในวง LAN (ไม่ได้เชื่อมต่อออกไปจริง)"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("10.255.255.255", 1))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="เปิดเซิร์ฟเวอร์ทดสอบแอปมือถือ")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if not (APP_DIR / "index.html").exists():
        print(f"ไม่พบไฟล์แอปที่ {APP_DIR}")
        return 1

    handler = partial(AppRequestHandler, directory=str(APP_DIR))
    try:
        server = ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    except OSError as exc:
        print(f"เปิดพอร์ต {args.port} ไม่ได้: {exc}")
        print(f"ลองใช้พอร์ตอื่น เช่น  run serve --port 8080")
        return 1

    print("=" * 62)
    print("เซิร์ฟเวอร์ทดสอบแอปมือถือ")
    print("=" * 62)
    print(f"\n  บนคอมพิวเตอร์เครื่องนี้ : http://localhost:{args.port}")
    print(f"  จากเครื่องอื่นในวง LAN  : http://{local_ip()}:{args.port}")
    print("\n  หมายเหตุ: เปิดจากมือถือผ่านเลข IP ข้างบนจะ **ใช้กล้องไม่ได้**")
    print("            เพราะเบราว์เซอร์บังคับให้ต้องเป็น https ถ้าไม่ใช่ localhost")
    print("            ทดสอบบนมือถือจริงให้ใช้ GitHub Pages")
    print("\n  กด Ctrl+C เพื่อหยุด\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nหยุดเซิร์ฟเวอร์แล้ว")
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
