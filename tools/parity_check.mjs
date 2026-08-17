/**
 * ฝั่ง JavaScript ของการตรวจสอบความตรงกัน (parity check)
 *
 * อ่านไฟล์ fixtures ที่ Python สร้างไว้ คำนวณ feature vector ด้วยโค้ดของแอปมือถือ
 * แล้วเขียนผลออกไปให้ Python เอาไปเทียบ  รันผ่าน `run parity` ไม่ต้องเรียกตรง ๆ
 */

import { readFileSync, writeFileSync } from "node:fs";
import {
  FEATURE_DIMS,
  buildFeatureVector,
  mirrorFeatureVector,
} from "../app/js/features.js";

const [inputPath, outputPath] = process.argv.slice(2);
if (!inputPath || !outputPath) {
  console.error("usage: node parity_check.mjs <fixtures.json> <results.json>");
  process.exit(2);
}

const fixtures = JSON.parse(readFileSync(inputPath, "utf8"));

const results = fixtures.cases.map((testCase) => {
  const hands = testCase.hands.map((hand) => ({
    handedness: hand.handedness,
    landmarks: hand.landmarks.map(([x, y, z]) => ({ x, y, z })),
  }));

  const vector = buildFeatureVector(hands);
  return {
    name: testCase.name,
    vector: Array.from(vector),
    mirrored: Array.from(mirrorFeatureVector(vector)),
  };
});

writeFileSync(
  outputPath,
  JSON.stringify({ featureDims: FEATURE_DIMS, results }, null, 1),
  "utf8"
);

console.log(`คำนวณด้วย JavaScript แล้ว ${results.length} กรณี -> ${outputPath}`);
