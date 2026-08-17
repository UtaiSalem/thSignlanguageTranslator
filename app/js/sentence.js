/**
 * รวมผลทายรายเฟรมให้กลายเป็นประโยค — ฉบับ JavaScript ของ SentenceBuilder
 * ใน src/translate.py
 *
 * กล้องส่งภาพมาราว 30 เฟรมต่อวินาที ถ้ารับทุกผลที่ทายได้ ข้อความจะกลายเป็น
 * "สวัสดีสวัสดีสวัสดี..." ภายในวินาทีเดียว จึงต้องมีตัวกรองสามชั้น:
 *   1. ความมั่นใจ — ทิ้งผลที่โมเดลไม่มั่นใจพอ
 *   2. ความนิ่ง   — ต้องทายคำเดิมติดกันหลายเฟรม
 *   3. ช่วงพัก    — หลังรับคำแล้วไม่รับคำเดิมซ้ำทันที
 */

export const DEFAULT_SETTINGS = {
  confidenceThreshold: 0.75,
  stableFramesRequired: 8,
  cooldownMs: 1000,
};

export class SentenceBuilder {
  constructor(settings = {}) {
    this.settings = { ...DEFAULT_SETTINGS, ...settings };
    this.words = [];
    this.recent = [];
    this.lastAccepted = null;
    this.acceptedAt = 0;
  }

  updateSettings(settings) {
    this.settings = { ...this.settings, ...settings };
    this.recent = [];
  }

  /**
   * ป้อนผลทายของเฟรมนี้ คืนคำที่เพิ่งถูกรับเข้าประโยค หรือ null ถ้ายังไม่รับ
   */
  update(label, confidence) {
    const { confidenceThreshold, stableFramesRequired, cooldownMs } = this.settings;

    // ชั้นที่ 1: ความมั่นใจ
    if (label === null || label === undefined || confidence < confidenceThreshold) {
      this.recent.length = 0;
      return null;
    }

    this.recent.push(label);
    if (this.recent.length > stableFramesRequired) this.recent.shift();

    // ชั้นที่ 2: ต้องนิ่งครบจำนวนเฟรม และเป็นคำเดียวกันทั้งหมด
    if (this.recent.length < stableFramesRequired) return null;
    for (const word of this.recent) {
      if (word !== label) return null;
    }

    // ชั้นที่ 3: ช่วงพักของคำเดิม
    const now = Date.now();
    if (label === this.lastAccepted && now - this.acceptedAt < cooldownMs) return null;

    this.words.push(label);
    this.lastAccepted = label;
    this.acceptedAt = now;
    this.recent.length = 0;
    return label;
  }

  /** สัดส่วนความนิ่งที่สะสมไว้ 0-1 ใช้วาดแถบความคืบหน้าให้ผู้ใช้เห็น */
  get progress() {
    return this.recent.length / this.settings.stableFramesRequired;
  }

  get text() {
    return this.words.join(" ");
  }

  addSpace() {
    if (this.words.length > 0 && this.words[this.words.length - 1] !== "|") {
      this.words.push("|");
    }
  }

  backspace() {
    this.words.pop();
    this.recent.length = 0;
    this.lastAccepted = null;
  }

  clear() {
    this.words.length = 0;
    this.recent.length = 0;
    this.lastAccepted = null;
  }

  /** ข้อความสำหรับแสดงผลและคัดลอก โดยแปลงเครื่องหมายเว้นวรรคให้อ่านง่าย */
  get displayText() {
    return this.words.join(" ").replace(/\|/g, "  ");
  }
}
