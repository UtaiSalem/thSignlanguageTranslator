/**
 * ทดสอบโมเดล MLP ฝั่ง JavaScript ด้วยข้อมูลจำลอง
 *
 * รัน:  run jstest
 *
 * ตอบคำถามว่า "โมเดลที่เขียนเองด้วย JavaScript เทรนได้จริงไหม เร็วแค่ไหน
 * และเซฟ/โหลดกลับมาแล้วได้ผลเหมือนเดิมหรือเปล่า" — ตรวจได้โดยไม่ต้องเปิดมือถือ
 */

import {
  FEATURE_DIMS,
  buildFeatureVector,
  mirrorFeatureVector,
} from "../app/js/features.js";
import { Classifier } from "../app/js/mlp.js";

function makeRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const labels = ["สวัสดี", "ขอบคุณ", "ใช่", "ไม่", "รัก"];
const SAMPLES_PER_CLASS = 80;

// สร้างชุดข้อมูลจำลอง: แต่ละคำมีท่าหลักหนึ่งท่า แล้วสั่นเล็กน้อยให้เหมือนเก็บจริง
const X = [];
const y = [];

for (let cls = 0; cls < labels.length; cls++) {
  const poseRandom = makeRandom(1000 + cls);
  const makePose = () =>
    Array.from({ length: 21 }, () => ({
      x: poseRandom() * 0.3 + 0.35,
      y: poseRandom() * 0.3 + 0.35,
      z: poseRandom() * 0.1,
    }));

  const base = makePose();
  const twoHanded = cls % 2 === 1; // สลับให้บางคำใช้สองมือ เหมือนภาษามือไทยจริง
  const secondHand = makePose();

  for (let rep = 0; rep < SAMPLES_PER_CLASS; rep++) {
    const jitter = makeRandom(cls * 7000 + rep);
    const noisy = (points) =>
      points.map((p) => ({
        x: p.x + (jitter() - 0.5) * 0.02,
        y: p.y + (jitter() - 0.5) * 0.02,
        z: p.z + (jitter() - 0.5) * 0.02,
      }));

    const hands = [{ handedness: "Right", landmarks: noisy(base) }];
    if (twoHanded) hands.push({ handedness: "Left", landmarks: noisy(secondHand) });

    X.push(buildFeatureVector(hands));
    y.push(cls);
  }
}

console.log("=".repeat(62));
console.log("ทดสอบโมเดล MLP ฝั่ง JavaScript");
console.log("=".repeat(62));
console.log(`\nชุดข้อมูลจำลอง: ${X.length} ตัวอย่าง, ${FEATURE_DIMS} มิติ, ${labels.length} คำ`);

const augmentedX = [...X, ...X.map(mirrorFeatureVector)];
const augmentedY = [...y, ...y];
console.log(`หลังเพิ่มข้อมูลแบบกลับด้าน: ${augmentedX.length} ตัวอย่าง\n`);

const model = new Classifier(labels, FEATURE_DIMS, 64);
const startedAt = Date.now();
let epochsRun = 0;

for (const progress of model.trainSteps(augmentedX, augmentedY, { epochs: 120, patience: 20 })) {
  epochsRun = progress.epoch;
  if (progress.epoch % 20 === 0 || progress.epoch === 1) {
    console.log(
      `  รอบที่ ${String(progress.epoch).padStart(3)}: ` +
        `loss=${progress.loss.toFixed(4)}  ` +
        `ความแม่นยำ=${(progress.accuracy * 100).toFixed(1)}%`
    );
  }
}

const elapsed = (Date.now() - startedAt) / 1000;
console.log(`\nเทรน ${epochsRun} รอบ ใช้เวลา ${elapsed.toFixed(2)} วินาที`);
console.log(`ความแม่นยำบนชุดตรวจสอบ: ${(model.accuracy * 100).toFixed(1)}%`);

let correct = 0;
for (let i = 0; i < X.length; i++) {
  if (model.predict(X[i]).label === labels[y[i]]) correct++;
}
const accuracy = correct / X.length;
console.log(`ความแม่นยำบนข้อมูลตั้งต้น: ${(accuracy * 100).toFixed(1)}%`);

const prediction = model.predict(X[0]);
const probabilitySum = prediction.probabilities.reduce((a, b) => a + b, 0);
console.log(
  `ทำนายตัวอย่างแรก: ${prediction.label} ที่ความมั่นใจ ` +
    `${(prediction.confidence * 100).toFixed(1)}%  (ผลรวมความน่าจะเป็น = ${probabilitySum.toFixed(6)})`
);

const serialised = JSON.parse(JSON.stringify(model.toJSON()));
const restored = Classifier.fromJSON(serialised);
let agreements = 0;
for (let i = 0; i < X.length; i++) {
  if (restored.predict(X[i]).label === model.predict(X[i]).label) agreements++;
}
const sizeKB = JSON.stringify(serialised).length / 1024;
console.log(`เซฟแล้วโหลดกลับ: ผลตรงกัน ${agreements}/${X.length} ตัวอย่าง (ไฟล์โมเดล ${sizeKB.toFixed(0)} KB)`);

const passed =
  accuracy > 0.95 && agreements === X.length && Math.abs(probabilitySum - 1) < 1e-5;

console.log();
console.log("=".repeat(62));
console.log(passed ? "ผ่านทุกข้อ — โมเดลฝั่งมือถือใช้งานได้" : "ไม่ผ่าน — ตรวจสอบ app/js/mlp.js");
console.log("=".repeat(62));

process.exit(passed ? 0 : 1);
