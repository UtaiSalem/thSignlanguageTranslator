/**
 * จัดการกล้องของมือถือ
 *
 * รองรับการสลับกล้องหน้า/หลัง เพราะสองโหมดของแอปใช้กล้องคนละตัว:
 *   - โหมดบันทึกท่า  ใช้กล้องหน้า (ทำท่าเองแล้วดูจอไปด้วย)
 *   - โหมดแปล       ใช้กล้องหลัง (ส่องไปที่คนที่กำลังทำท่า)
 */

export class Camera {
  constructor(videoElement) {
    this.video = videoElement;
    this.stream = null;
    this.facingMode = "user"; // "user" = กล้องหน้า, "environment" = กล้องหลัง
  }

  get isMirrored() {
    // กล้องหน้าควรแสดงแบบกระจกเงา ไม่งั้นผู้ใช้จะขยับมือผิดทางโดยสัญชาตญาณ
    return this.facingMode === "user";
  }

  async start(facingMode = this.facingMode) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error(
        "เบราว์เซอร์นี้เปิดกล้องไม่ได้ — ต้องเปิดผ่าน https หรือ localhost เท่านั้น"
      );
    }

    this.stop();
    this.facingMode = facingMode;

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: facingMode },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });
    } catch (error) {
      if (error.name === "NotAllowedError") {
        throw new Error("ไม่ได้รับอนุญาตให้ใช้กล้อง — กดอนุญาตในแถบที่อยู่ของเบราว์เซอร์");
      }
      if (error.name === "NotFoundError") {
        throw new Error("ไม่พบกล้องในเครื่องนี้");
      }
      throw error;
    }

    this.video.srcObject = this.stream;
    await this.video.play();

    // รอให้ได้ขนาดภาพจริงก่อน ไม่งั้น MediaPipe จะได้เฟรมขนาด 0x0
    if (this.video.videoWidth === 0) {
      await new Promise((resolve) => {
        this.video.addEventListener("loadedmetadata", resolve, { once: true });
      });
    }

    return this;
  }

  async switchCamera() {
    // ต้องจำค่าเดิมไว้ก่อน เพราะ start() ตั้ง this.facingMode เป็นค่าใหม่ตั้งแต่ต้น
    // ถ้าไปอ่าน this.facingMode ในบล็อก catch จะได้ค่าใหม่ที่เพิ่งล้มเหลวไป
    // แล้วกู้กลับไปกล้องตัวที่ใช้ไม่ได้ซ้ำอีกรอบ
    const previous = this.facingMode;
    const next = previous === "user" ? "environment" : "user";

    try {
      await this.start(next);
    } catch (error) {
      // มือถือบางรุ่นมีกล้องเดียว ให้กลับไปใช้ตัวเดิมแทนที่จะพังทั้งแอป
      try {
        await this.start(previous);
      } catch {
        // กู้กลับไม่ได้ด้วย ปล่อยให้ผู้ใช้เห็นข้อความเดียวกันแล้วรีเฟรชหน้าเว็บเอง
      }
      throw new Error("สลับกล้องไม่ได้ — เครื่องนี้อาจมีกล้องตัวเดียว");
    }
    return this.facingMode;
  }

  stop() {
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
      this.stream = null;
    }
    this.video.srcObject = null;
  }

  get isRunning() {
    return this.stream !== null;
  }
}
