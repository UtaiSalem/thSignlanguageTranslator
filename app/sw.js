/**
 * Service worker — ทำให้แอปเปิดใช้งานได้แม้ไม่มีอินเทอร์เน็ต
 *
 * เก็บทุกอย่างที่จำเป็นลงแคชตั้งแต่ตอนติดตั้ง รวมถึงไฟล์ก้อนใหญ่สองไฟล์
 * (ตัวประมวลผล WASM 11 MB และโมเดลตรวจจับมือ 7.8 MB) การติดตั้งครั้งแรก
 * จึงใช้เน็ตพอสมควร แต่หลังจากนั้นเปิดใช้งานแบบออฟไลน์ได้เลย
 *
 * กลยุทธ์การแคชแยกตามชนิดไฟล์ เพราะสองกลุ่มนี้มีความต้องการต่างกันคนละขั้ว:
 *
 *   ไฟล์เล็ก (html/css/js/manifest/icon) — stale-while-revalidate
 *       ส่งของจากแคชทันทีเพื่อให้เปิดแอปเร็วและใช้ออฟไลน์ได้ แล้วแอบโหลดรุ่นใหม่
 *       มาทับแคชเบื้องหลัง การเปิดครั้งถัดไปจะได้รุ่นใหม่เอง
 *
 *   ไฟล์ก้อนใหญ่ (.wasm, .task รวมกัน 19 MB) — cache-first ไม่ตรวจซ้ำ
 *       ถ้าตรวจซ้ำทุกครั้งจะกินเน็ตมือถือมหาศาลโดยไม่ได้อะไร ไฟล์กลุ่มนี้จะได้
 *       รุ่นใหม่เมื่อเพิ่มเลข CACHE_VERSION ซึ่งล้างแคชเก่าทั้งชุด
 *
 * เดิมทั้งสองกลุ่มเป็น cache-first หมด ผู้ใช้ที่ติดตั้งแอปไว้แล้วจึงไม่ได้รับ
 * การแก้ไขใด ๆ เลยจนกว่าจะมีคนจำได้ว่าต้องเพิ่มเลข CACHE_VERSION ด้วยมือ
 */

const CACHE_VERSION = "v2";
const CACHE_NAME = `sign-translator-${CACHE_VERSION}`;

const ASSETS = [
  "./",
  "./index.html",
  "./manifest.webmanifest",
  "./css/style.css",
  "./js/app.js",
  "./js/camera.js",
  "./js/features.js",
  "./js/handTracker.js",
  "./js/mlp.js",
  "./js/sentence.js",
  "./js/storage.js",
  "./js/transfer.js",
  "./vendor/vision_bundle.mjs",
  "./vendor/wasm/vision_wasm_internal.js",
  "./vendor/wasm/vision_wasm_internal.wasm",
  "./models/hand_landmarker.task",
  "./icons/icon-192.png",
  "./icons/icon-512.png",
  "./icons/icon-maskable-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE_NAME);
      // ใส่ทีละไฟล์แทน cache.addAll เพราะถ้าไฟล์เดียวพลาด addAll จะล้มทั้งชุด
      // แล้วแอปจะใช้งานออฟไลน์ไม่ได้เลยโดยไม่มีอะไรบอก
      await Promise.all(
        ASSETS.map(async (url) => {
          try {
            await cache.add(new Request(url, { cache: "reload" }));
          } catch (error) {
            console.warn("แคชไฟล์นี้ไม่สำเร็จ:", url, error);
          }
        })
      );
      await self.skipWaiting();
    })()
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const names = await caches.keys();
      await Promise.all(
        names.filter((name) => name !== CACHE_NAME).map((name) => caches.delete(name))
      );
      await self.clients.claim();
    })()
  );
});

/** ไฟล์ก้อนใหญ่ที่ไม่คุ้มจะตรวจหารุ่นใหม่ทุกครั้งที่เปิดแอป */
function isLargeBinary(url) {
  return /\.(wasm|task)$/.test(new URL(url).pathname);
}

/** เก็บลงแคชเฉพาะคำตอบที่ใช้ได้จริง — กัน error page ทับของดีในแคช */
function isCacheable(response) {
  return response && response.ok && response.type === "basic";
}

async function cacheFirst(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request, { ignoreSearch: true });
  if (cached) return cached;

  const response = await fetch(request);
  if (isCacheable(response)) await cache.put(request, response.clone());
  return response;
}

async function staleWhileRevalidate(request, event) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request, { ignoreSearch: true });

  const fromNetwork = fetch(request)
    .then(async (response) => {
      if (isCacheable(response)) await cache.put(request, response.clone());
      return response;
    })
    .catch(() => null);

  if (cached) {
    // ผู้ใช้ได้ของจากแคชทันที ส่วนการโหลดรุ่นใหม่ปล่อยให้วิ่งต่อเบื้องหลัง
    // ต้องบอก waitUntil ไว้ ไม่งั้นเบราว์เซอร์อาจฆ่า service worker ก่อนโหลดเสร็จ
    event.waitUntil(fromNetwork);
    return cached;
  }

  const response = await fromNetwork;
  if (response) return response;

  // ออฟไลน์และไม่มีในแคช — ถ้าเป็นการเปิดหน้าเว็บ ให้ย้อนกลับไปหน้าแรก
  if (request.mode === "navigate") {
    const fallback = await cache.match("./index.html");
    if (fallback) return fallback;
  }
  return Response.error();
}

self.addEventListener("fetch", (event) => {
  const request = event.request;

  // จัดการเฉพาะการโหลดหน้าเว็บและไฟล์ของแอปเท่านั้น
  if (request.method !== "GET") return;
  if (!request.url.startsWith(self.location.origin)) return;

  event.respondWith(
    isLargeBinary(request.url) ? cacheFirst(request) : staleWhileRevalidate(request, event)
  );
});
