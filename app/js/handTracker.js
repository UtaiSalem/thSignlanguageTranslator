/**
 * ห่อ MediaPipe HandLandmarker ฝั่งเว็บ ให้มีหน้าตาเหมือน src/hand_tracker.py
 *
 * ไฟล์ WASM และไฟล์โมเดลถูกเก็บไว้ในโปรเจคนี้ทั้งหมด (โฟลเดอร์ vendor/ และ models/)
 * ไม่ได้ดึงจาก CDN เพราะแอปต้องทำงานได้แม้ไม่มีอินเทอร์เน็ต
 */

import {
  FilesetResolver,
  HandLandmarker,
} from "../vendor/vision_bundle.mjs";

export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

export class HandTracker {
  constructor() {
    this.landmarker = null;
    this.lastTimestamp = -1;
  }

  async load({ maxHands = 2, onProgress = () => {} } = {}) {
    onProgress("กำลังเตรียมตัวประมวลผล ...");
    const fileset = await FilesetResolver.forVisionTasks("./vendor/wasm");

    onProgress("กำลังโหลดโมเดลตรวจจับมือ (7.8 MB) ...");
    this.landmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: "./models/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: maxHands,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    onProgress("พร้อมใช้งาน");
    return this;
  }

  /**
   * ตรวจจับมือจากเฟรมวิดีโอ
   * @param {HTMLVideoElement} video
   * @param {number} timestampMs ต้องเพิ่มขึ้นเสมอ ไม่งั้น MediaPipe จะ error
   * @returns {Array<{landmarks:Array, handedness:string, score:number}>}
   */
  detect(video, timestampMs) {
    if (!this.landmarker) return [];

    // โหมด VIDEO บังคับว่า timestamp ต้องมากขึ้นเรื่อย ๆ กันไว้ตรงนี้เลย
    let timestamp = Math.round(timestampMs);
    if (timestamp <= this.lastTimestamp) timestamp = this.lastTimestamp + 1;
    this.lastTimestamp = timestamp;

    const result = this.landmarker.detectForVideo(video, timestamp);
    const hands = [];

    for (let i = 0; i < result.landmarks.length; i++) {
      const category = result.handednesses[i][0];
      hands.push({
        landmarks: result.landmarks[i],
        handedness: category.categoryName,
        score: category.score,
      });
    }

    return hands;
  }

  close() {
    if (this.landmarker) {
      this.landmarker.close();
      this.landmarker = null;
    }
  }
}

/** คำนวณกรอบสี่เหลี่ยมล้อมมือ เป็นพิกัดพิกเซล */
export function handBoundingBox(hand, width, height, padding = 16) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const point of hand.landmarks) {
    const x = point.x * width;
    const y = point.y * height;
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }

  return {
    x: Math.max(minX - padding, 0),
    y: Math.max(minY - padding, 0),
    width: Math.min(maxX + padding, width) - Math.max(minX - padding, 0),
    height: Math.min(maxY + padding, height) - Math.max(minY - padding, 0),
  };
}
