/**
 * นำข้อมูลเข้า-ออก เพื่อแชร์ระหว่างมือถือกับคอมพิวเตอร์
 *
 * ใช้ JSON เพราะอ่านออกด้วยตาเปล่าและทั้งสองฝั่งจัดการได้ง่าย
 * ฝั่งคอมพิวเตอร์แปลงไป-กลับกับไฟล์ CSV ด้วย `run transfer` (ดู src/transfer.py)
 *
 * สิ่งที่แชร์กันคือ **ชุดข้อมูล** ไม่ใช่ตัวโมเดล เพราะสองฝั่งใช้คนละไลบรารี
 * แต่ feature vector 128 ค่าเหมือนกันทุกประการ (ตรวจสอบด้วย `run parity`)
 * แต่ละฝั่งจึงเทรนโมเดลของตัวเองจากข้อมูลชุดเดียวกันได้
 */

import { FEATURE_DIMS } from "./features.js";

export const DATASET_FORMAT = "sign-language-translator-dataset";
export const DATASET_VERSION = 1;

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

  if (data.featureDims !== FEATURE_DIMS) {
    throw new Error(
      `จำนวนมิติของข้อมูลไม่ตรงกัน (ไฟล์มี ${data.featureDims} แอปต้องการ ${FEATURE_DIMS})`
    );
  }

  if (!Array.isArray(data.samples) || data.samples.length === 0) {
    throw new Error("ไฟล์นี้ไม่มีข้อมูลตัวอย่างอยู่เลย");
  }

  const samples = [];
  for (const [index, sample] of data.samples.entries()) {
    if (typeof sample.label !== "string" || !sample.label) {
      throw new Error(`ตัวอย่างลำดับที่ ${index + 1} ไม่มีชื่อคำกำกับ`);
    }
    if (!Array.isArray(sample.vector) || sample.vector.length !== FEATURE_DIMS) {
      throw new Error(`ตัวอย่างลำดับที่ ${index + 1} มีจำนวนค่าไม่ครบ ${FEATURE_DIMS}`);
    }
    samples.push({ label: sample.label, vector: Float32Array.from(sample.vector) });
  }

  return { samples, source: data.source || "unknown", exportedAt: data.exportedAt };
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
