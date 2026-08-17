/**
 * นำข้อมูลเข้า-ออก เพื่อแชร์ระหว่างมือถือกับคอมพิวเตอร์
 *
 * ใช้ JSON เพราะอ่านออกด้วยตาเปล่าและทั้งสองฝั่งจัดการได้ง่าย
 * ฝั่งคอมพิวเตอร์แปลงไป-กลับกับไฟล์ CSV ด้วย `run transfer` (ดู src/transfer.py)
 *
 * สิ่งที่แชร์กันคือ **ชุดข้อมูล** ไม่ใช่ตัวโมเดล เพราะสองฝั่งใช้คนละไลบรารี
 * แต่ feature vector 133 ค่าเหมือนกันทุกประการ (ตรวจสอบด้วย `run parity`)
 * แต่ละฝั่งจึงเทรนโมเดลของตัวเองจากข้อมูลชุดเดียวกันได้
 */

import {
  FEATURE_DIMS,
  LEGACY_FEATURE_DIMS,
  mirrorFeatureVector,
  upgradeLegacyVector,
} from "./features.js";

export const DATASET_FORMAT = "sign-language-translator-dataset";
export const DATASET_VERSION = 3;

/**
 * ไฟล์รุ่นเก่ายังนำเข้าได้ แต่ต้องแปลงก่อน มีสองเรื่องที่ต่างกันและเป็นอิสระจากกัน
 *
 *   รุ่น 1 — 128 มิติ และฝั่งคอมพิวเตอร์ยังเก็บในคอนเวนชันภาพกระจก
 *   รุ่น 2 — 128 มิติ คอนเวนชันมือตรงกันแล้ว แต่ยังไม่มีท่อนคู่มือ
 *   รุ่น 3 — 133 มิติ มีท่อนคู่มือ (ดู features.js)
 */
export const LEGACY_VERSIONS = [1, 2];
export const MIRRORED_CONVENTION_VERSION = 1;

/** สร้างเนื้อหาไฟล์ JSON จากตัวอย่างทั้งหมด */
export function buildExport(samples) {
  return {
    format: DATASET_FORMAT,
    version: DATASET_VERSION,
    featureDims: FEATURE_DIMS,
    exportedAt: new Date().toISOString(),
    source: "mobile",
    sampleCount: samples.length,
    samples: samples.map((sample) => ({
      label: sample.label,
      vector: Array.from(sample.vector).map((value) => Number(value.toFixed(6))),
    })),
  };
}

/** สั่งให้เบราว์เซอร์ดาวน์โหลดไฟล์ */
export function downloadJSON(data, filename) {
  const blob = new Blob([JSON.stringify(data)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  // ปล่อยหน่วยความจำคืน แต่ต้องรอให้เบราว์เซอร์เริ่มดาวน์โหลดก่อน
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * ตรวจสอบและอ่านไฟล์ที่นำเข้ามา
 * โยน Error พร้อมข้อความภาษาไทยถ้าไฟล์ผิดรูปแบบ
 */
export function parseImport(text) {
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error("ไฟล์นี้ไม่ใช่ JSON ที่ถูกต้อง");
  }

  if (data.format !== DATASET_FORMAT) {
    throw new Error("ไม่ใช่ไฟล์ชุดข้อมูลของแอปนี้");
  }

  const isLegacy = LEGACY_VERSIONS.includes(data.version);
  if (!isLegacy && data.version !== DATASET_VERSION) {
    throw new Error(
      `ไฟล์นี้เป็นรุ่น ${data.version} ซึ่งแอปรุ่นนี้อ่านไม่ได้ (อ่านได้ถึงรุ่น ${DATASET_VERSION})`
    );
  }

  const expectedDims = isLegacy ? LEGACY_FEATURE_DIMS : FEATURE_DIMS;
  if (data.featureDims !== expectedDims) {
    throw new Error(
      `จำนวนมิติของข้อมูลไม่ตรงกัน (ไฟล์รุ่น ${data.version} มี ${data.featureDims} ควรเป็น ${expectedDims})`
    );
  }

  if (!Array.isArray(data.samples) || data.samples.length === 0) {
    throw new Error("ไฟล์นี้ไม่มีข้อมูลตัวอย่างอยู่เลย");
  }

  const source = data.source || "unknown";

  // ไฟล์รุ่น 1 จากคอมพิวเตอร์อยู่ในคอนเวนชันภาพกระจก ต้องสลับก่อนเอามาปนกับข้อมูลใหม่
  const needsMirror = data.version === MIRRORED_CONVENTION_VERSION && source === "desktop";

  const samples = [];
  for (const [index, sample] of data.samples.entries()) {
    if (typeof sample.label !== "string" || !sample.label) {
      throw new Error(`ตัวอย่างลำดับที่ ${index + 1} ไม่มีชื่อคำกำกับ`);
    }
    if (!Array.isArray(sample.vector) || sample.vector.length !== expectedDims) {
      throw new Error(`ตัวอย่างลำดับที่ ${index + 1} มีจำนวนค่าไม่ครบ ${expectedDims}`);
    }
    let vector = Float32Array.from(sample.vector);
    if (isLegacy) vector = upgradeLegacyVector(vector);
    if (needsMirror) vector = mirrorFeatureVector(vector);
    samples.push({ label: sample.label, vector });
  }

  return {
    samples,
    source,
    exportedAt: data.exportedAt,
    converted: isLegacy,
    mirrored: needsMirror,
  };
}

/** เปิดหน้าต่างเลือกไฟล์แล้วคืนเนื้อหาเป็นข้อความ */
export function pickFile(accept = "application/json,.json") {
  return new Promise((resolve, reject) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = accept;

    input.onchange = () => {
      const file = input.files && input.files[0];
      if (!file) {
        resolve(null);
        return;
      }
      const reader = new FileReader();
      reader.onload = () => resolve({ name: file.name, text: reader.result });
      reader.onerror = () => reject(new Error("อ่านไฟล์ไม่สำเร็จ"));
      reader.readAsText(file);
    };

    input.click();
  });
}
