/**
 * Service worker — ทำให้แอปเปิดใช้งานได้แม้ไม่มีอินเทอร์เน็ต
 *
 * เก็บทุกอย่างที่จำเป็นลงแคชตั้งแต่ตอนติดตั้ง รวมถึงไฟล์ก้อนใหญ่สองไฟล์
 * (ตัวประมวลผล WASM 11 MB และโมเดลตรวจจับมือ 7.8 MB) การติดตั้งครั้งแรก
 * จึงใช้เน็ตพอสมควร แต่หลังจากนั้นเปิดใช้งานแบบออฟไลน์ได้เลย
 *
 * **เมื่อแก้ไฟล์ใด ๆ ในโฟลเดอร์ app/ ต้องเพิ่มเลข CACHE_VERSION ด้วย**
 * ไม่งั้นเบราว์เซอร์จะยังเสิร์ฟไฟล์เก่าจากแคชต่อไป
 */

const CACHE_VERSION = "v1";
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

self.addEventListener("fetch", (event) => {
  const request = event.request;

  // จัดการเฉพาะการโหลดหน้าเว็บและไฟล์ของแอปเท่านั้น
  if (request.method !== "GET") return;
  if (!request.url.startsWith(self.location.origin)) return;

  event.respondWith(
    (async () => {
      const cached = await caches.match(request, { ignoreSearch: true });
      if (cached) return cached;

      try {
        const response = await fetch(request);
        // เก็บไฟล์ที่โหลดสำเร็จเพิ่มเข้าแคชไว้ใช้ครั้งหน้า
        if (response.ok && response.type === "basic") {
          const cache = await caches.open(CACHE_NAME);
          cache.put(request, response.clone());
        }
        return response;
      } catch (error) {
        // ออฟไลน์และไม่มีในแคช — ถ้าเป็นการเปิดหน้าเว็บ ให้ย้อนกลับไปหน้าแรก
        if (request.mode === "navigate") {
          const fallback = await caches.match("./index.html");
          if (fallback) return fallback;
        }
        throw error;
      }
    })()
  );
});
