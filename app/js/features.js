/**
 * แปลงจุด landmark ของมือให้เป็นเวกเตอร์คุณลักษณะ 133 ค่า
 *
 * ไฟล์นี้เป็นฉบับ JavaScript ของ src/features.py และ **ต้องให้ผลตรงกันทุกหลัก**
 * เพราะข้อมูลที่เก็บจากมือถือกับจากคอมพิวเตอร์ต้องสลับไปมาได้
 * ถ้าแก้ไฟล์นี้ ต้องแก้ features.py ให้ตรงกันเสมอ แล้วรัน `run parity` เพื่อตรวจสอบ
 *
 * **คอนเวนชันกลาง:** เวกเตอร์ทุกตัวอ้างอิงกับ "ภาพดิบจากกล้อง (ยังไม่พลิกกระจก)"
 * ฝั่งนี้ส่งภาพดิบจาก <video> เข้า MediaPipe ตรง ๆ (การพลิกกระจกของกล้องหน้าทำด้วย
 * CSS transform ตอนแสดงผลเท่านั้น ไม่ได้แตะข้อมูลที่ส่งเข้าโมเดล) เวกเตอร์ที่ได้จึงอยู่
 * ในคอนเวนชันกลางอยู่แล้ว ไม่ต้องแปลงอะไร ส่วนฝั่งคอมพิวเตอร์ที่พลิกเฟรมก่อนตรวจจับ
 * ต้องแปลงกลับด้วย build_canonical_feature_vector (ดู src/features.py)
 *
 * ดูคำอธิบายว่าทำไมต้อง normalize ได้ที่ src/features.py และ docs/
 */

export const NUM_LANDMARKS = 21;
export const COORDS_PER_LANDMARK = 3;
export const PER_HAND_DIMS = NUM_LANDMARKS * COORDS_PER_LANDMARK + 1; // 64
export const HAND_SLOTS = ["Left", "Right"];
export const HANDS_DIMS = PER_HAND_DIMS * HAND_SLOTS.length; // 128

// ท่อนคู่มือ: การ normalize เทียบข้อมือของแต่ละมือเอง ทำให้ความสัมพันธ์ระหว่าง
// สองมือหายไปหมด (มือประกบกันกับมือแยกห่างกันได้ตัวเลขชุดเดียวกัน) อีก 5 ค่านี้
// เก็บส่วนที่หายไปกลับมา — ดูคำอธิบายเต็มที่ src/features.py
export const PAIR_DIMS = 5;
export const PAIR_START = HANDS_DIMS;
export const FEATURE_DIMS = HANDS_DIMS + PAIR_DIMS; // 133

/** จำนวนมิติของชุดข้อมูลรุ่นก่อนที่จะมีท่อนคู่มือ ใช้ตอนนำเข้าข้อมูลเก่า */
export const LEGACY_FEATURE_DIMS = HANDS_DIMS;

/**
 * normalize จุดของมือหนึ่งข้าง คืนอาเรย์ 63 ค่า
 * @param {Array<{x:number,y:number,z:number}>} landmarks จุด 21 จุดจาก MediaPipe
 * @returns {Float32Array}
 */
export function normalizeLandmarks(landmarks) {
  return normalizeWithGeometry(landmarks).normalized;
}

/**
 * เหมือน normalizeLandmarks แต่คืนตำแหน่งข้อมือและสเกลที่หารทิ้งไปด้วย
 * สองค่านี้คือสิ่งที่ท่อนคู่มือต้องใช้
 * @param {Array<{x:number,y:number,z:number}>} landmarks
 * @returns {{normalized: Float32Array, wrist: {x:number,y:number,z:number}, scale: number}}
 */
export function normalizeWithGeometry(landmarks) {
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

  // ปัดสเกลเป็น float32 ก่อนคืนออกไป เพราะฝั่ง Python เก็บค่านี้เป็น float32
  // ถ้าปล่อยเป็น float64 ท่อนคู่มือของสองฝั่งจะต่างกันเกินเกณฑ์ที่ `run parity` ยอมรับ
  return { normalized: centered, wrist, scale: Math.fround(scale) };
}

/**
 * รวมมือที่ตรวจเจอ (0-2 ข้าง) เป็นเวกเตอร์ความยาวคงที่ 133 ค่า
 *
 * โครงของเวกเตอร์:
 *   [  0..63 ]  มือ Left  : 63 ค่า normalize แล้ว + ธงบอกการมีอยู่
 *   [ 64..127]  มือ Right : เหมือนกัน
 *   [128..132]  ท่อนคู่มือ : offset x/y/z, สมดุลขนาด, ธงบอกว่าค่าใช้ได้
 *
 * @param {Array<{landmarks:Array, handedness:string}>} hands
 * @returns {Float32Array}
 */
export function buildFeatureVector(hands) {
  const vector = new Float32Array(FEATURE_DIMS);
  const wrists = new Array(HAND_SLOTS.length).fill(null);
  const scales = new Array(HAND_SLOTS.length).fill(0);

  for (const hand of hands) {
    const slot = HAND_SLOTS.indexOf(hand.handedness);
    if (slot < 0) continue;

    const start = slot * PER_HAND_DIMS;
    const { normalized, wrist, scale } = normalizeWithGeometry(hand.landmarks);
    vector.set(normalized, start);
    vector[start + PER_HAND_DIMS - 1] = 1.0; // ธง: มือข้างนี้ปรากฏอยู่
    wrists[slot] = wrist;
    scales[slot] = scale;
  }

  if (wrists.every((wrist) => wrist !== null)) fillPairBlock(vector, wrists, scales);

  return vector;
}

/**
 * เติมท่อนคู่มือ — เรียกเฉพาะเมื่อเจอมือครบสองข้างเท่านั้น
 *
 * offset หารด้วยขนาดมือเฉลี่ย เพื่อให้ยังไม่ขึ้นกับระยะห่างจากกล้อง
 * สมดุลขนาดใช้ (ขวา-ซ้าย)/(ขวา+ซ้าย) ซึ่งอยู่ในช่วง -1 ถึง 1 เสมอ
 * ดูเหตุผลของทั้งสองสูตรที่ src/features.py
 */
function fillPairBlock(vector, wrists, scales) {
  const [leftWrist, rightWrist] = wrists;
  const [leftScale, rightScale] = scales;

  let meanScale = Math.fround((leftScale + rightScale) / 2);
  if (meanScale < 1e-6) meanScale = 1.0;

  vector[PAIR_START] = (rightWrist.x - leftWrist.x) / meanScale;
  vector[PAIR_START + 1] = (rightWrist.y - leftWrist.y) / meanScale;
  vector[PAIR_START + 2] = (rightWrist.z - leftWrist.z) / meanScale;

  const totalScale = Math.fround(leftScale + rightScale);
  if (totalScale > 1e-6) {
    vector[PAIR_START + 3] = (rightScale - leftScale) / totalScale;
  }

  vector[PAIR_START + 4] = 1.0; // ธง: ค่าในท่อนนี้วัดมาจริง ไม่ใช่ศูนย์เพราะไม่รู้
}

/**
 * ขยายเวกเตอร์รุ่นเก่า 128 ค่า ให้เป็น 133 ค่า
 *
 * ท่อนคู่มือของข้อมูลเก่ากู้กลับไม่ได้จริง ๆ จึงเติมศูนย์แล้วปล่อยธงเป็น 0
 * ซึ่งหมายถึง "ไม่รู้ค่า" ไม่ใช่ "สองมือซ้อนทับกันพอดี" (ดู src/features.py)
 * @param {Float32Array|Array<number>} values
 * @returns {Float32Array}
 */
export function upgradeLegacyVector(values) {
  const upgraded = new Float32Array(FEATURE_DIMS);
  for (let i = 0; i < LEGACY_FEATURE_DIMS; i++) upgraded[i] = values[i];
  return upgraded;
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

  // ท่อนคู่มือกลับด้าน "ตรงข้าม" กับท่อนของแต่ละมือ: แกน x คงเดิม แกน y กับ z สลับ
  // เพราะการสลับช่องซ้าย-ขวากลับทิศของตัวลบเอง ซึ่งหักล้างกับการพลิกภาพบนแกน x พอดี
  // ดูการพิสูจน์ทีละบรรทัดที่ src/features.py
  mirrored[PAIR_START] = vector[PAIR_START];
  mirrored[PAIR_START + 1] = -vector[PAIR_START + 1];
  mirrored[PAIR_START + 2] = -vector[PAIR_START + 2];
  mirrored[PAIR_START + 3] = -vector[PAIR_START + 3];
  mirrored[PAIR_START + 4] = vector[PAIR_START + 4];

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
  names.push(
    "Pair_offset_x",
    "Pair_offset_y",
    "Pair_offset_z",
    "Pair_scale_balance",
    "Pair_present"
  );
  return names;
}
