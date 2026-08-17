/**
 * แปลงจุด landmark ของมือให้เป็นเวกเตอร์คุณลักษณะ 128 ค่า
 *
 * ไฟล์นี้เป็นฉบับ JavaScript ของ src/features.py และ **ต้องให้ผลตรงกันทุกหลัก**
 * เพราะข้อมูลที่เก็บจากมือถือกับจากคอมพิวเตอร์ต้องสลับไปมาได้
 * ถ้าแก้ไฟล์นี้ ต้องแก้ features.py ให้ตรงกันเสมอ แล้วรัน `run parity` เพื่อตรวจสอบ
 *
 * ดูคำอธิบายว่าทำไมต้อง normalize ได้ที่ src/features.py และ docs/
 */

export const NUM_LANDMARKS = 21;
export const COORDS_PER_LANDMARK = 3;
export const PER_HAND_DIMS = NUM_LANDMARKS * COORDS_PER_LANDMARK + 1; // 64
export const HAND_SLOTS = ["Left", "Right"];
export const FEATURE_DIMS = PER_HAND_DIMS * HAND_SLOTS.length; // 128

/**
 * normalize จุดของมือหนึ่งข้าง คืนอาเรย์ 63 ค่า
 * @param {Array<{x:number,y:number,z:number}>} landmarks จุด 21 จุดจาก MediaPipe
 * @returns {Float32Array}
 */
export function normalizeLandmarks(landmarks) {
  const wrist = landmarks[0];

  // 1) ย้ายข้อมือไปที่จุดกำเนิด — ตัดผลของตำแหน่งในเฟรม
  const centered = new Float32Array(NUM_LANDMARKS * COORDS_PER_LANDMARK);
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    const p = landmarks[i];
    centered[i * 3] = p.x - wrist.x;
    centered[i * 3 + 1] = p.y - wrist.y;
    centered[i * 3 + 2] = p.z - wrist.z;
  }

  // 2) หารด้วยระยะที่ไกลที่สุดจากข้อมือ — ตัดผลของขนาด/ระยะห่างจากกล้อง
  let scale = 0;
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    const x = centered[i * 3];
    const y = centered[i * 3 + 1];
    const z = centered[i * 3 + 2];
    const distance = Math.hypot(x, y, z);
    if (distance > scale) scale = distance;
  }
  if (scale < 1e-6) scale = 1.0; // กันหารด้วยศูนย์

  for (let i = 0; i < centered.length; i++) centered[i] /= scale;
  return centered;
}

/**
 * รวมมือที่ตรวจเจอ (0-2 ข้าง) เป็นเวกเตอร์ความยาวคงที่ 128 ค่า
 * @param {Array<{landmarks:Array, handedness:string}>} hands
 * @returns {Float32Array}
 */
export function buildFeatureVector(hands) {
  const vector = new Float32Array(FEATURE_DIMS);

  for (const hand of hands) {
    const slot = HAND_SLOTS.indexOf(hand.handedness);
    if (slot < 0) continue;

    const start = slot * PER_HAND_DIMS;
    const normalized = normalizeLandmarks(hand.landmarks);
    vector.set(normalized, start);
    vector[start + PER_HAND_DIMS - 1] = 1.0; // ธง: มือข้างนี้ปรากฏอยู่
  }

  return vector;
}

/**
 * สร้างเวกเตอร์ของภาพสะท้อนกระจก ใช้เพิ่มข้อมูลตอนเทรน
 * กลับเครื่องหมายแกน x และสลับช่องมือซ้าย-ขวาพร้อมกัน
 * @param {Float32Array|Array<number>} vector
 * @returns {Float32Array}
 */
export function mirrorFeatureVector(vector) {
  const mirrored = new Float32Array(FEATURE_DIMS);

  for (let slot = 0; slot < HAND_SLOTS.length; slot++) {
    const start = slot * PER_HAND_DIMS;
    const present = vector[start + PER_HAND_DIMS - 1];
    const other = (slot + 1) % HAND_SLOTS.length;
    const dest = other * PER_HAND_DIMS;

    for (let i = 0; i < NUM_LANDMARKS; i++) {
      mirrored[dest + i * 3] = -vector[start + i * 3]; // กลับแกน x
      mirrored[dest + i * 3 + 1] = vector[start + i * 3 + 1];
      mirrored[dest + i * 3 + 2] = vector[start + i * 3 + 2];
    }
    mirrored[dest + PER_HAND_DIMS - 1] = present;
  }

  return mirrored;
}

/** ชื่อคอลัมน์ ใช้ตอน export เป็น CSV ให้ตรงกับฝั่ง Python */
export function featureColumnNames() {
  const names = [];
  for (const slot of HAND_SLOTS) {
    for (let i = 0; i < NUM_LANDMARKS; i++) {
      names.push(`${slot}_${i}_x`, `${slot}_${i}_y`, `${slot}_${i}_z`);
    }
    names.push(`${slot}_present`);
  }
  return names;
}
