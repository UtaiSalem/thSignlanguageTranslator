/**
 * ตัวควบคุมหลักของแอป — ผูกทุกส่วนเข้าด้วยกันและจัดการหน้าจอทั้งสามโหมด
 *
 *   โหมดบันทึก  เปิดกล้อง ทำท่า แล้วบอกว่าท่านี้แปลว่าอะไร
 *   โหมดเทรน    เทรนโมเดลจากข้อมูลที่เก็บไว้ และนำข้อมูลเข้า-ออก
 *   โหมดแปล     ส่องกล้องไปที่ผู้ทำท่า แล้วอ่านคำแปลบนหน้าจอ
 */

import { Camera } from "./camera.js";
import { FEATURE_DIMS, buildFeatureVector, mirrorFeatureVector } from "./features.js";
import { HAND_CONNECTIONS, HandTracker, handBoundingBox } from "./handTracker.js";
import { Classifier } from "./mlp.js";
import { DEFAULT_SETTINGS, SentenceBuilder } from "./sentence.js";
import * as storage from "./storage.js";
import * as transfer from "./transfer.js";

const $ = (id) => document.getElementById(id);

const state = {
  tracker: null,
  camera: null,
  model: null,
  builder: new SentenceBuilder(),
  mode: "collect",
  running: false,
  burst: false,
  lastBurstAt: 0,
  currentLabel: "",
  lastHands: [],      // ผลตรวจจับของเฟรมล่าสุด ใช้ตอนกดปุ่มเก็บตัวอย่าง
  counts: new Map(),
  flashUntil: 0,
  frameTimes: [],
};

const BURST_INTERVAL_MS = 120;

/* ------------------------------------------------------------------ ตัวช่วย */

function toast(message, kind = "info") {
  const element = $("toast");
  element.textContent = message;
  element.className = `toast toast--${kind} toast--visible`;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => {
    element.className = "toast";
  }, 3200);
}

/** คืนการควบคุมให้เบราว์เซอร์ชั่วขณะ เพื่อไม่ให้หน้าจอค้างระหว่างงานหนัก */
const yieldToUI = () => new Promise((resolve) => setTimeout(resolve, 0));

function setStatus(text) {
  $("loadingText").textContent = text;
}

/* ---------------------------------------------------------- การวาดภาพซ้อน */

function drawOverlay(hands) {
  const canvas = $("overlay");
  const video = $("video");
  const context = canvas.getContext("2d");

  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }

  context.clearRect(0, 0, canvas.width, canvas.height);
  const scale = Math.max(canvas.width / 640, 1);

  for (const hand of hands) {
    const points = hand.landmarks.map((p) => [p.x * canvas.width, p.y * canvas.height]);

    context.strokeStyle = "rgba(226, 232, 240, 0.85)";
    context.lineWidth = 3 * scale;
    context.beginPath();
    for (const [start, end] of HAND_CONNECTIONS) {
      context.moveTo(points[start][0], points[start][1]);
      context.lineTo(points[end][0], points[end][1]);
    }
    context.stroke();

    context.fillStyle = "#38bdf8";
    for (const [x, y] of points) {
      context.beginPath();
      context.arc(x, y, 4.5 * scale, 0, Math.PI * 2);
      context.fill();
    }

    const box = handBoundingBox(hand, canvas.width, canvas.height);
    context.strokeStyle = "rgba(34, 197, 94, 0.9)";
    context.lineWidth = 3 * scale;
    context.strokeRect(box.x, box.y, box.width, box.height);
  }
}

/* --------------------------------------------------------------- ลูปหลัก */

function loop() {
  if (!state.running) return;
  requestAnimationFrame(loop);

  const video = $("video");
  if (!video.videoWidth) return;

  const now = performance.now();
  state.frameTimes.push(now);
  if (state.frameTimes.length > 30) state.frameTimes.shift();

  let hands = [];
  try {
    hands = state.tracker.detect(video, now);
  } catch (error) {
    console.error("ตรวจจับมือผิดพลาด", error);
    return;
  }
  state.lastHands = hands;

  drawOverlay(hands);

  if (state.mode === "collect") updateCollect(hands, now);
  else if (state.mode === "translate") updateTranslate(hands);
}

function currentFps() {
  const times = state.frameTimes;
  if (times.length < 2) return 0;
  return ((times.length - 1) * 1000) / (times[times.length - 1] - times[0]);
}

/* ------------------------------------------------------------ โหมดบันทึก */

async function captureSample(hands) {
  if (!hands.length) {
    toast("ยังไม่เห็นมือในกรอบภาพ", "warn");
    return false;
  }
  if (!state.currentLabel) {
    toast("พิมพ์ก่อนว่าท่านี้แปลว่าอะไร", "warn");
    return false;
  }

  await storage.addSample(state.currentLabel, buildFeatureVector(hands));
  state.counts.set(state.currentLabel, (state.counts.get(state.currentLabel) || 0) + 1);
  state.flashUntil = performance.now() + 140;
  renderLabelList();
  return true;
}

function updateCollect(hands, now) {
  $("collectHandInfo").textContent = hands.length
    ? `เห็นมือ ${hands.length} ข้าง`
    : "ยกมือเข้ามาในกรอบ";
  $("collectHandInfo").dataset.ok = hands.length > 0 ? "yes" : "no";

  const count = state.counts.get(state.currentLabel) || 0;
  $("collectCount").textContent = state.currentLabel
    ? `"${state.currentLabel}" เก็บแล้ว ${count} ตัวอย่าง`
    : "ยังไม่ได้ตั้งชื่อท่า";

  if (state.burst && hands.length && now - state.lastBurstAt >= BURST_INTERVAL_MS) {
    state.lastBurstAt = now;
    captureSample(hands);
  }

  $("viewport").classList.toggle("viewport--flash", now < state.flashUntil);
}

async function renderLabelList() {
  state.counts = await storage.countByLabel();
  const container = $("labelList");
  const labels = [...state.counts.keys()].sort((a, b) => a.localeCompare(b, "th"));

  const total = [...state.counts.values()].reduce((a, b) => a + b, 0);
  $("datasetSummary").textContent = labels.length
    ? `${labels.length} คำ · ${total} ตัวอย่าง`
    : "ยังไม่มีข้อมูล";

  if (!labels.length) {
    container.innerHTML = '<p class="hint">ยังไม่มีคำที่บันทึกไว้</p>';
    return;
  }

  container.innerHTML = "";
  for (const label of labels) {
    const count = state.counts.get(label);
    const row = document.createElement("div");
    row.className = "label-row";
    if (label === state.currentLabel) row.classList.add("label-row--active");

    const name = document.createElement("button");
    name.className = "label-row__name";
    name.textContent = label;
    name.onclick = () => {
      state.currentLabel = label;
      $("labelInput").value = label;
      renderLabelList();
    };

    const badge = document.createElement("span");
    badge.className = "label-row__count";
    badge.textContent = count;
    // 150 ตัวอย่างขึ้นไปถือว่าเพียงพอ ให้สัญญาณสีบอกผู้ใช้
    badge.dataset.level = count >= 150 ? "good" : count >= 60 ? "ok" : "low";

    const remove = document.createElement("button");
    remove.className = "label-row__delete";
    remove.textContent = "ลบ";
    remove.onclick = async () => {
      if (!confirm(`ลบข้อมูลทั้งหมดของคำว่า "${label}" ใช่ไหม`)) return;
      const removed = await storage.deleteLabel(label);
      toast(`ลบ "${label}" แล้ว ${removed} ตัวอย่าง`);
      if (state.currentLabel === label) state.currentLabel = "";
      renderLabelList();
    };

    row.append(name, badge, remove);
    container.appendChild(row);
  }
}

/* -------------------------------------------------------------- โหมดเทรน */

async function trainModel() {
  const samples = await storage.getAllSamples();
  const labels = [...new Set(samples.map((s) => s.label))].sort((a, b) => a.localeCompare(b, "th"));

  if (labels.length < 2) {
    toast("ต้องมีอย่างน้อย 2 คำถึงจะเทรนได้", "warn");
    return;
  }

  const counts = new Map();
  for (const sample of samples) counts.set(sample.label, (counts.get(sample.label) || 0) + 1);
  const fewest = Math.min(...counts.values());
  if (fewest < 10) {
    const weakest = [...counts.entries()].find(([, c]) => c === fewest)[0];
    toast(`คำว่า "${weakest}" มีแค่ ${fewest} ตัวอย่าง ควรเก็บเพิ่ม`, "warn");
  }

  const labelIndex = new Map(labels.map((label, index) => [label, index]));
  const X = samples.map((s) => Float32Array.from(s.vector));
  const y = samples.map((s) => labelIndex.get(s.label));

  const panel = $("trainProgress");
  panel.hidden = false;
  $("trainButton").disabled = true;

  const model = new Classifier(labels, FEATURE_DIMS, 64);
  const startedAt = Date.now();
  let lastYield = startedAt;

  // ส่งข้อมูลดิบเข้าไป แล้วให้ trainSteps เพิ่มข้อมูลแบบกลับด้านซ้าย-ขวาเองหลังแบ่ง
  // ชุดตรวจสอบแล้ว (ถ้าเพิ่มมาก่อน ความแม่นยำที่รายงานจะสูงเกินจริง)
  // การเพิ่มข้อมูลนี้ทำให้โมเดลไม่ยึดติดว่าต้องใช้มือข้างไหน และรองรับกรณี
  // เก็บข้อมูลด้วยกล้องหน้าแต่ไปใช้งานด้วยกล้องหลัง
  const trainOptions = { epochs: 150, patience: 25, augment: mirrorFeatureVector };

  for (const progress of model.trainSteps(X, y, trainOptions)) {
    $("trainBar").style.width = `${(progress.epoch / progress.epochs) * 100}%`;
    $("trainStat").textContent =
      `รอบที่ ${progress.epoch}/${progress.epochs} · ` +
      `ความแม่นยำ ${(progress.accuracy * 100).toFixed(1)}%`;

    // คืนการควบคุมให้หน้าจอเป็นระยะ ไม่งั้นแอปจะค้างจนกว่าจะเทรนเสร็จ
    if (Date.now() - lastYield > 60) {
      lastYield = Date.now();
      await yieldToUI();
    }
  }

  const elapsed = ((Date.now() - startedAt) / 1000).toFixed(1);
  state.model = model;
  await storage.setMeta("model", model.toJSON());

  $("trainBar").style.width = "100%";
  $("trainStat").textContent =
    `เสร็จแล้ว · ความแม่นยำ ${(model.accuracy * 100).toFixed(1)}% · ใช้เวลา ${elapsed} วินาที`;
  $("trainButton").disabled = false;

  toast(`เทรนเสร็จ ความแม่นยำ ${(model.accuracy * 100).toFixed(1)}%`, "ok");
  renderModelInfo();
}

function renderModelInfo() {
  const info = $("modelInfo");
  if (!state.model) {
    info.innerHTML = '<p class="hint">ยังไม่มีโมเดล — กดปุ่มเทรนด้านบน</p>';
    $("translateHint").hidden = false;
    return;
  }

  const when = state.model.trainedAt
    ? new Date(state.model.trainedAt).toLocaleString("th-TH")
    : "ไม่ทราบเวลา";
  info.innerHTML = `
    <p><strong>${state.model.labels.length} คำ</strong> ·
       ความแม่นยำ ${(state.model.accuracy * 100).toFixed(1)}%</p>
    <p class="hint">เทรนเมื่อ ${when}</p>
    <p class="chips">${state.model.labels.map((l) => `<span class="chip">${l}</span>`).join("")}</p>
  `;
  $("translateHint").hidden = true;
}

/* --------------------------------------------------------------- โหมดแปล */

function updateTranslate(hands) {
  if (!state.model) return;

  let label = null;
  let confidence = 0;
  let probabilities = null;

  if (hands.length) {
    const result = state.model.predict(buildFeatureVector(hands));
    label = result.label;
    confidence = result.confidence;
    probabilities = result.probabilities;
  }

  const accepted = state.builder.update(label, confidence);
  if (accepted) {
    state.flashUntil = performance.now() + 260;
    renderSentence();
    if ($("speakToggle").checked) speak(accepted);
  }

  const threshold = state.builder.settings.confidenceThreshold;
  const wordElement = $("predictedWord");
  const confidenceElement = $("predictedConfidence");

  if (!hands.length) {
    wordElement.textContent = "ยกมือเข้ามาในกรอบ";
    wordElement.dataset.state = "idle";
    confidenceElement.textContent = "";
  } else if (confidence < threshold) {
    wordElement.textContent = "ไม่แน่ใจ";
    wordElement.dataset.state = "unsure";
    confidenceElement.textContent = `ความมั่นใจสูงสุด ${(confidence * 100).toFixed(0)}%`;
  } else {
    wordElement.textContent = label;
    wordElement.dataset.state = "sure";
    confidenceElement.textContent = `ความมั่นใจ ${(confidence * 100).toFixed(0)}%`;
  }

  $("stabilityBar").style.width = `${state.builder.progress * 100}%`;
  $("fpsLabel").textContent = `${currentFps().toFixed(0)} FPS`;

  renderTopPredictions(probabilities);
  $("viewport").classList.toggle("viewport--flash", performance.now() < state.flashUntil);
}

function renderTopPredictions(probabilities) {
  const container = $("topPredictions");
  if (!probabilities || !state.model) {
    container.innerHTML = "";
    return;
  }

  const ranked = probabilities
    .map((value, index) => ({ label: state.model.labels[index], value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 3);

  container.innerHTML = ranked
    .map(
      (item, rank) => `
      <div class="prediction${rank === 0 ? " prediction--top" : ""}">
        <span class="prediction__label">${item.label}</span>
        <span class="prediction__bar"><i style="width:${item.value * 100}%"></i></span>
      </div>`
    )
    .join("");
}

function renderSentence() {
  const text = state.builder.displayText;
  $("sentence").textContent = text || "—";
  $("sentence").dataset.empty = text ? "no" : "yes";
}

/** อ่านออกเสียงภาษาไทย ใช้เสียงสังเคราะห์ที่มีอยู่ในเครื่อง */
function speak(text) {
  if (!("speechSynthesis" in window)) return;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "th-TH";
  utterance.rate = 0.95;
  speechSynthesis.speak(utterance);
}

/* ------------------------------------------------------- การสลับโหมดหน้าจอ */

async function setMode(mode) {
  state.mode = mode;
  state.burst = false;
  $("burstButton").dataset.active = "no";

  for (const name of ["collect", "train", "translate"]) {
    $(`panel-${name}`).hidden = name !== mode;
    $(`tab-${name}`).dataset.active = name === mode ? "yes" : "no";
  }

  const needsCamera = mode === "collect" || mode === "translate";
  $("viewport").hidden = !needsCamera;
  $("viewport").dataset.mode = mode; // CSS ใช้ค่านี้ตัดสินว่าจะแสดงป้ายผลการแปลไหม

  if (needsCamera) {
    // โหมดบันทึกใช้กล้องหน้า (ทำท่าเอง) โหมดแปลใช้กล้องหลัง (ส่องคนอื่น)
    const desired = mode === "collect" ? "user" : "environment";
    if (!state.camera.isRunning) {
      await startCamera(desired);
    } else if (state.camera.facingMode !== desired && mode === "translate") {
      // ไม่บังคับสลับ ปล่อยให้ผู้ใช้กดเองด้วยปุ่มสลับกล้อง
    }
    state.running = true;
    loop();
  } else {
    state.running = false;
  }

  if (mode === "translate") {
    if (!state.model) toast("ยังไม่มีโมเดล — ไปที่แท็บเทรนก่อน", "warn");
    state.builder.clear();
    renderSentence();
  }
  if (mode === "train") renderLabelList();
}

async function startCamera(facingMode) {
  try {
    await state.camera.start(facingMode);
    applyMirror();
  } catch (error) {
    toast(error.message, "error");
    throw error;
  }
}

function applyMirror() {
  const mirrored = state.camera.isMirrored;
  $("viewport").dataset.mirrored = mirrored ? "yes" : "no";
  $("cameraLabel").textContent = state.camera.facingMode === "user" ? "กล้องหน้า" : "กล้องหลัง";
}

/* ------------------------------------------------------------ นำข้อมูลเข้า-ออก */

async function exportDataset() {
  const samples = await storage.getAllSamples();
  if (!samples.length) {
    toast("ยังไม่มีข้อมูลให้ส่งออก", "warn");
    return;
  }
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  transfer.downloadJSON(transfer.buildExport(samples), `signdata-${stamp}.json`);
  toast(`ส่งออก ${samples.length} ตัวอย่างแล้ว`, "ok");
}

async function importDataset() {
  let picked;
  try {
    picked = await transfer.pickFile();
  } catch (error) {
    toast(error.message, "error");
    return;
  }
  if (!picked) return;

  let parsed;
  try {
    parsed = transfer.parseImport(picked.text);
  } catch (error) {
    toast(error.message, "error");
    return;
  }

  await storage.addSamples(parsed.samples);
  const note = parsed.converted ? " (แปลงจากไฟล์รุ่นเก่าให้แล้ว)" : "";
  toast(`นำเข้า ${parsed.samples.length} ตัวอย่างแล้ว${note}`, "ok");
  renderLabelList();
}

async function clearDataset() {
  if (!confirm("ลบข้อมูลที่เก็บไว้ทั้งหมด ใช่ไหม การกระทำนี้ย้อนกลับไม่ได้")) return;
  await storage.clearAllSamples();
  state.currentLabel = "";
  $("labelInput").value = "";
  toast("ล้างข้อมูลทั้งหมดแล้ว");
  renderLabelList();
}

/* --------------------------------------------------------------- ตั้งค่า */

function bindSettings() {
  const inputs = {
    confidence: $("settingConfidence"),
    stability: $("settingStability"),
    cooldown: $("settingCooldown"),
  };

  const apply = async () => {
    const settings = {
      confidenceThreshold: Number(inputs.confidence.value) / 100,
      stableFramesRequired: Number(inputs.stability.value),
      cooldownMs: Number(inputs.cooldown.value),
    };
    state.builder.updateSettings(settings);
    await storage.setMeta("settings", settings);

    $("settingConfidenceValue").textContent = `${inputs.confidence.value}%`;
    $("settingStabilityValue").textContent = `${inputs.stability.value} เฟรม`;
    $("settingCooldownValue").textContent = `${(Number(inputs.cooldown.value) / 1000).toFixed(1)} วินาที`;
  };

  for (const input of Object.values(inputs)) input.oninput = apply;
  return { inputs, apply };
}

/* ------------------------------------------------------------------ เริ่มต้น */

async function main() {
  const settingsUI = bindSettings();
  let pendingNotice = null;

  // โหลดการตั้งค่าที่เคยบันทึกไว้
  const savedSettings = await storage.getMeta("settings", DEFAULT_SETTINGS);
  state.builder.updateSettings(savedSettings);
  settingsUI.inputs.confidence.value = Math.round(savedSettings.confidenceThreshold * 100);
  settingsUI.inputs.stability.value = savedSettings.stableFramesRequired;
  settingsUI.inputs.cooldown.value = savedSettings.cooldownMs;
  await settingsUI.apply();

  // โหลดโมเดลที่เคยเทรนไว้
  const savedModel = await storage.getMeta("model");
  if (savedModel) {
    // โมเดลที่เทรนไว้ก่อนเพิ่มท่อนคู่มือรับ input คนละขนาด ถ้าปล่อยให้โหลดต่อ
    // มันจะอ่านแค่ 128 ค่าแรกแล้วทายต่อได้เงียบ ๆ ซึ่งซ่อนปัญหาไว้ ทิ้งแล้วบอกให้เทรนใหม่
    if (savedModel.inputDim !== FEATURE_DIMS) {
      await storage.setMeta("model", null);
      // เก็บข้อความไว้แสดงหลังหน้าจอโหลดหายไป ไม่งั้นผู้ใช้จะไม่ได้เห็น
      pendingNotice = { text: "โมเดลเดิมใช้กับข้อมูลรุ่นใหม่ไม่ได้ — กดเทรนใหม่ที่แท็บเทรน", kind: "warn" };
    } else {
      try {
        state.model = Classifier.fromJSON(savedModel);
      } catch (error) {
        console.error("โหลดโมเดลที่บันทึกไว้ไม่สำเร็จ", error);
      }
    }
  }
  renderModelInfo();
  await renderLabelList();

  state.camera = new Camera($("video"));
  state.tracker = new HandTracker();

  try {
    await state.tracker.load({ onProgress: setStatus });
  } catch (error) {
    console.error(error);
    setStatus("โหลดโมเดลตรวจจับมือไม่สำเร็จ — ลองรีเฟรชหน้าเว็บ");
    return;
  }

  setStatus("กำลังขอสิทธิ์ใช้กล้อง ...");
  try {
    await startCamera("user");
  } catch {
    setStatus("เปิดกล้องไม่ได้ — ตรวจสอบการอนุญาตแล้วรีเฟรชหน้าเว็บ");
    return;
  }

  $("loading").hidden = true;
  $("app").hidden = false;

  await setMode("collect");

  if (pendingNotice) toast(pendingNotice.text, pendingNotice.kind);
}

/* ------------------------------------------------------------ ผูกปุ่มต่าง ๆ */

function bindControls() {
  $("tab-collect").onclick = () => setMode("collect");
  $("tab-train").onclick = () => setMode("train");
  $("tab-translate").onclick = () => setMode("translate");

  $("labelInput").oninput = (event) => {
    state.currentLabel = event.target.value.trim();
  };

  $("captureButton").onclick = async () => {
    // ใช้ผลตรวจจับของเฟรมล่าสุดที่ลูปทำไว้แล้ว การเรียก detect() ซ้ำตรงนี้
    // คือการประมวลผลเฟรมเดิมสองรอบ ซึ่งเปลืองแรงเครื่องเปล่า ๆ บนมือถือ
    await captureSample(state.lastHands);
  };

  $("burstButton").onclick = (event) => {
    state.burst = !state.burst;
    event.currentTarget.dataset.active = state.burst ? "yes" : "no";
    event.currentTarget.textContent = state.burst ? "หยุดเก็บต่อเนื่อง" : "เก็บต่อเนื่อง";
  };

  $("undoButton").onclick = async () => {
    if (!state.currentLabel) return;
    const removed = await storage.deleteLastSample(state.currentLabel);
    toast(removed ? "ลบตัวอย่างล่าสุดแล้ว" : "ไม่มีตัวอย่างให้ลบ", removed ? "info" : "warn");
    renderLabelList();
  };

  $("switchCameraButton").onclick = async () => {
    try {
      await state.camera.switchCamera();
      applyMirror();
    } catch (error) {
      toast(error.message, "warn");
    }
  };

  $("trainButton").onclick = () => trainModel().catch((error) => {
    console.error(error);
    toast("เทรนไม่สำเร็จ ดูรายละเอียดใน console", "error");
    $("trainButton").disabled = false;
  });

  $("exportButton").onclick = exportDataset;
  $("importButton").onclick = importDataset;
  $("clearButton").onclick = clearDataset;

  $("spaceButton").onclick = () => {
    state.builder.addSpace();
    renderSentence();
  };
  $("backspaceButton").onclick = () => {
    state.builder.backspace();
    renderSentence();
  };
  $("clearSentenceButton").onclick = () => {
    state.builder.clear();
    renderSentence();
  };
  $("copyButton").onclick = async () => {
    const text = state.builder.displayText;
    if (!text) {
      toast("ยังไม่มีข้อความให้คัดลอก", "warn");
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
      toast("คัดลอกข้อความแล้ว", "ok");
    } catch {
      toast("คัดลอกไม่สำเร็จ", "error");
    }
  };
  $("speakButton").onclick = () => {
    const text = state.builder.displayText;
    if (text) speak(text);
  };

  // หยุดกล้องเมื่อสลับไปแอปอื่น เพื่อประหยัดแบตเตอรี่และความเป็นส่วนตัว
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      state.running = false;
      state.burst = false;
      $("burstButton").dataset.active = "no";
      $("burstButton").textContent = "เก็บต่อเนื่อง";
    } else if (state.mode !== "train" && state.camera && state.camera.isRunning) {
      state.running = true;
      loop();
    }
  });
}

bindControls();
main().catch((error) => {
  console.error(error);
  setStatus(`เกิดข้อผิดพลาด: ${error.message}`);
});
