/**
 * เก็บข้อมูลลงเครื่อง ด้วย IndexedDB
 *
 * ทุกอย่างอยู่ในเครื่องผู้ใช้ทั้งหมด ไม่มีการส่งข้อมูลออกไปที่เซิร์ฟเวอร์ใด ๆ
 * ซึ่งสำคัญมากเพราะข้อมูลที่เก็บมาจากกล้องของผู้ใช้เอง
 *
 * เลือก IndexedDB แทน localStorage เพราะ localStorage เก็บได้แค่ราว 5 MB
 * และเก็บได้เฉพาะข้อความ ส่วนชุดข้อมูลของเราเป็นตัวเลขหลักแสนค่า
 */

const DB_NAME = "sign-language-translator";
const DB_VERSION = 1;
const STORE_SAMPLES = "samples";
const STORE_META = "meta";

let dbPromise = null;

function openDatabase() {
  if (dbPromise) return dbPromise;

  dbPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains(STORE_SAMPLES)) {
        const store = db.createObjectStore(STORE_SAMPLES, {
          keyPath: "id",
          autoIncrement: true,
        });
        store.createIndex("label", "label", { unique: false });
      }
      if (!db.objectStoreNames.contains(STORE_META)) {
        db.createObjectStore(STORE_META, { keyPath: "key" });
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });

  return dbPromise;
}

function runTransaction(storeName, mode, callback) {
  return openDatabase().then(
    (db) =>
      new Promise((resolve, reject) => {
        const transaction = db.transaction(storeName, mode);
        const store = transaction.objectStore(storeName);
        let result;
        try {
          result = callback(store);
        } catch (error) {
          reject(error);
          return;
        }
        transaction.oncomplete = () => resolve(result && result.__request ? result.__request.result : result);
        transaction.onerror = () => reject(transaction.error);
        transaction.onabort = () => reject(transaction.error);
      })
  );
}

/** เพิ่มตัวอย่างหนึ่งรายการ */
export async function addSample(label, vector) {
  return runTransaction(STORE_SAMPLES, "readwrite", (store) => {
    store.add({
      label,
      vector: Array.from(vector),
      createdAt: Date.now(),
    });
  });
}

/** เพิ่มหลายรายการพร้อมกัน ใช้ตอนนำเข้าไฟล์ เร็วกว่าเรียกทีละรายการมาก */
export async function addSamples(records) {
  return runTransaction(STORE_SAMPLES, "readwrite", (store) => {
    for (const record of records) {
      store.add({
        label: record.label,
        vector: Array.from(record.vector),
        createdAt: record.createdAt || Date.now(),
      });
    }
  });
}

/** อ่านตัวอย่างทั้งหมด */
export async function getAllSamples() {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const request = db.transaction(STORE_SAMPLES, "readonly").objectStore(STORE_SAMPLES).getAll();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

/** นับจำนวนตัวอย่างของแต่ละคำ คืน Map ของ ชื่อคำ -> จำนวน */
export async function countByLabel() {
  const samples = await getAllSamples();
  const counts = new Map();
  for (const sample of samples) {
    counts.set(sample.label, (counts.get(sample.label) || 0) + 1);
  }
  return counts;
}

/** ลบตัวอย่างล่าสุดของคำที่ระบุ คืน true ถ้าลบสำเร็จ */
export async function deleteLastSample(label) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_SAMPLES, "readwrite");
    const store = transaction.objectStore(STORE_SAMPLES);
    const request = store.index("label").openCursor(IDBKeyRange.only(label), "prev");

    request.onsuccess = () => {
      const cursor = request.result;
      if (!cursor) {
        resolve(false);
        return;
      }
      cursor.delete();
      resolve(true);
    };
    request.onerror = () => reject(request.error);
  });
}

/** ลบตัวอย่างทั้งหมดของคำที่ระบุ คืนจำนวนที่ลบไป */
export async function deleteLabel(label) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_SAMPLES, "readwrite");
    const store = transaction.objectStore(STORE_SAMPLES);
    const request = store.index("label").openCursor(IDBKeyRange.only(label));
    let removed = 0;

    request.onsuccess = () => {
      const cursor = request.result;
      if (!cursor) return;
      cursor.delete();
      removed++;
      cursor.continue();
    };
    transaction.oncomplete = () => resolve(removed);
    transaction.onerror = () => reject(transaction.error);
  });
}

/** ล้างชุดข้อมูลทั้งหมด */
export async function clearAllSamples() {
  return runTransaction(STORE_SAMPLES, "readwrite", (store) => store.clear());
}

/** บันทึกค่าลงที่เก็บทั่วไป (ใช้เก็บโมเดลและการตั้งค่า) */
export async function setMeta(key, value) {
  return runTransaction(STORE_META, "readwrite", (store) => store.put({ key, value }));
}

/** อ่านค่าจากที่เก็บทั่วไป */
export async function getMeta(key, fallback = null) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const request = db.transaction(STORE_META, "readonly").objectStore(STORE_META).get(key);
    request.onsuccess = () => resolve(request.result ? request.result.value : fallback);
    request.onerror = () => reject(request.error);
  });
}

/** ประมาณพื้นที่ที่ใช้ไปและที่เหลือ (เบราว์เซอร์บางตัวไม่รองรับ) */
export async function estimateStorage() {
  if (!navigator.storage || !navigator.storage.estimate) return null;
  const { usage, quota } = await navigator.storage.estimate();
  return { usage, quota };
}
