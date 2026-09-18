"use strict";

const $ = (s) => document.querySelector(s);
const SVGNS = "http://www.w3.org/2000/svg";
const WINDOW_S = 300;
const SPARK_W = 200;
const SPARK_H = 40;

let llmList = [];
let selected = localStorage.getItem("llmrunner-model");
let pollingHw = false;
let pollingLlms = false;
let pollingMetrics = false;
let busy = false;

/* ---------- theme ---------- */
const rootEl = document.documentElement;
const stored = localStorage.getItem("llmrunner-theme");
if (stored) rootEl.dataset.theme = stored;
else if (matchMedia("(prefers-color-scheme: light)").matches) rootEl.dataset.theme = "light";
$("#themeToggle").addEventListener("click", () => {
  rootEl.dataset.theme = rootEl.dataset.theme === "light" ? "dark" : "light";
  localStorage.setItem("llmrunner-theme", rootEl.dataset.theme);
});

/* ---------- toast ---------- */
let toastTimer = null;
function toast(msg, isErr = false) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.toggle("err", isErr);
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 6000);
}

/* ---------- fetch helpers ---------- */
async function jget(url) {
  const r = await fetch(url);
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
  return d;
}
async function jpost(url) {
  const r = await fetch(url, { method: "POST" });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
  return d;
}

/* ---------- formatting ---------- */
const isNum = (v) => typeof v === "number" && isFinite(v);
const fmt = (v, digits = 1) => (isNum(v) ? v.toFixed(digits) : "n/a");
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function inWindow(pairs) {
  const now = Date.now() / 1000;
  return (pairs || []).filter((p) => p && isNum(p[1]) && now - p[0] <= WINDOW_S);
}
function scaleSeries(pairs, f) {
  return (pairs || []).map(([ts, v]) => [ts, isNum(v) ? v * f : null]);
}
function peak(pairs) {
  const v = inWindow(pairs).map((p) => p[1]);
  return v.length ? Math.max(...v) : null;
}

/* ---------- sparkline on a fixed 5-minute time axis ---------- */
function sparkPath(pairs, domain) {
  const pts = inWindow(pairs);
  if (pts.length < 2) return "";
  let lo, hi;
  if (domain) [lo, hi] = domain;
  else {
    const vs = pts.map((p) => p[1]);
    lo = Math.min(0, ...vs);
    hi = Math.max(...vs) * 1.2 || 1;
  }
  if (hi <= lo) hi = lo + 1;
  const t0 = Date.now() / 1000 - WINDOW_S;
  return pts.map(([ts, v], i) => {
    const x = Math.min(Math.max(((ts - t0) / WINDOW_S) * SPARK_W, 0), SPARK_W);
    const y = SPARK_H - 2 - ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * (SPARK_H - 4);
    return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
}

/* ---------- cards ---------- */
const CARDS = {
  sys: [
    { label: "CPU Util", color: "var(--accent)", domain: [0, 100],
      series: (d) => d.series.cpu_pct, value: (d) => fmt(d.sys.cpu_pct, 0), unit: " %",
      sub: (d) => `load ${fmt(d.sys.load_1m, 2)}` },
    { label: "CPU Freq", color: "var(--accent)",
      series: (d) => scaleSeries(d.series.cpu_freq_mhz, 1 / 1000),
      value: (d) => fmt(isNum(d.sys.cpu_freq_mhz) ? d.sys.cpu_freq_mhz / 1000 : null, 2), unit: " GHz",
      sub: (d) => (isNum(d.cpu_freq_max) ? `max ${(d.cpu_freq_max / 1000).toFixed(1)} GHz` : "") },
    { label: "CPU Temp", color: "var(--warn)", domain: [0, 100],
      series: (d) => d.series.cpu_temp_c, value: (d) => fmt(d.sys.cpu_temp_c, 0), unit: " °C",
      sub: (d) => { const p = peak(d.series.cpu_temp_c); return p === null ? "" : `peak ${p.toFixed(0)}°C`; } },
    { label: "Memory", color: "var(--accent)", domain: [0, 100],
      series: (d) => d.series.mem_pct, value: (d) => fmt(d.sys.mem_pct, 0), unit: " %",
      sub: (d) => `${fmt(d.sys.mem_used_gb, 1)} / ${fmt(d.sys.mem_total_gb, 0)} GiB` },
  ],
  gpu: [
    { label: "GPU Util", color: "var(--accent-2)", domain: [0, 100],
      series: (d) => d.gpu_series.util_pct, value: (d) => fmt(d.gpu.util_pct, 0), unit: " %",
      sub: (d) => d.gpu.name || "" },
    { label: "GPU Freq", color: "var(--accent)",
      series: (d) => d.gpu_series.clock_mhz, value: (d) => fmt(d.gpu.clock_mhz, 0), unit: " MHz",
      sub: (d) => { const p = peak(d.gpu_series.clock_mhz); return p === null ? "" : `max ${p.toFixed(0)} MHz`; } },
    { label: "GPU Temp", color: "var(--warn)", domain: [0, 100],
      series: (d) => d.gpu_series.temp_c, value: (d) => fmt(d.gpu.temp_c, 0), unit: " °C",
      sub: (d) => { const p = peak(d.gpu_series.temp_c); return p === null ? "" : `peak ${p.toFixed(0)}°C`; } },
    { label: "GPU Power", color: "var(--accent-2)",
      series: (d) => d.gpu_series.power_w, value: (d) => fmt(d.gpu.power_w, 0), unit: " W",
      sub: (d) => (isNum(d.gpu.power_limit_w) ? `limit ${d.gpu.power_limit_w.toFixed(0)} W` : "") },
  ],
  storage: [
    { label: "Storage /", color: "var(--accent)", domain: [0, 100],
      series: (d) => d.series.disk_pct, value: (d) => fmt(d.sys.disk_pct, 0), unit: " %",
      sub: (d) => `${fmt(d.sys.disk_used_gb, 0)} / ${fmt(d.sys.disk_total_gb, 0)} GiB` },
    { label: "Disk Read", color: "var(--accent)",
      series: (d) => d.series.disk_read_mbps, value: (d) => fmt(d.sys.disk_read_mbps, 1), unit: " MB/s", sub: () => "" },
    { label: "Disk Write", color: "var(--accent-2)",
      series: (d) => d.series.disk_write_mbps, value: (d) => fmt(d.sys.disk_write_mbps, 1), unit: " MB/s", sub: () => "" },
  ],
};

const cardRefs = {};
function buildCards(group, specs) {
  const row = $("#row" + group[0].toUpperCase() + group.slice(1));
  row.innerHTML = "";
  specs.forEach((spec, i) => {
    const card = document.createElement("section");
    card.className = "card";
    card.innerHTML =
      '<div class="head"><span class="label"></span><span class="sub"></span></div>' +
      '<div class="value"></div>' +
      `<svg class="spark" viewBox="0 0 ${SPARK_W} ${SPARK_H}" preserveAspectRatio="none"><path fill="none" stroke="${spec.color}" stroke-width="1.8" stroke-linejoin="round"/></svg>`;
    row.appendChild(card);
    const key = group + ":" + i;
    cardRefs[key] = {
      spec,
      label: card.querySelector(".label"),
      sub: card.querySelector(".sub"),
      value: card.querySelector(".value"),
      path: card.querySelector("path"),
    };
    cardRefs[key].label.textContent = spec.label;
  });
}
function updateCards(group, data) {
  (CARDS[group] || []).forEach((spec, i) => {
    const ref = cardRefs[group + ":" + i];
    if (!ref) return;
    let pairs = [];
    let latest = null;
    try {
      pairs = spec.series(data) || [];
      latest = spec.value(data);
    } catch (e) { /* missing gpu data */ }
    ref.value.innerHTML = `${latest ?? "n/a"}<small>${ref.spec.unit}</small>`;
    ref.sub.textContent = spec.sub ? spec.sub(data) : "";
    ref.path.setAttribute("d", sparkPath(pairs, spec.domain) || "M0,0");
  });
}

/* ---------- hero ---------- */
function byName(name) {
  return llmList.find((l) => l.name === name);
}
function heroState(llm) {
  if (llm.status === "running") return ["running", "RUNNING"];
  if (llm.status === "starting") return ["starting", "STARTING"];
  return ["stopped", "STOPPED"];
}
function renderLlms(data) {
  llmList = data.llms;
  if (!llmList.length) return;
  if (!byName(selected)) {
    selected = (llmList.find((l) => l.running) || llmList[0]).name;
    localStorage.setItem("llmrunner-model", selected);
  }

  renderLoadList($("#loadModal").hidden ? "" : $("#loadSearch").value);
  renderHero(byName(selected));
}
function renderLoadList(filter = "") {
  const list = $("#loadList");
  list.innerHTML = "";
  const q = filter.trim().toLowerCase();
  const frag = document.createDocumentFragment();
  llmList.forEach((llm) => {
    if (q && !(llm.name.toLowerCase().includes(q) || llm.type.toLowerCase().includes(q))) return;
    const item = document.createElement("button");
    item.type = "button";
    item.className = "load-item";
    const ep = llm.endpoint.replace(/^https?:\/\//, "") + " · " + llm.type;
    item.innerHTML =
      `<span class="dot ${llm.status === "running" ? "on" : ""}"></span>` +
      `<span class="name">${esc(llm.name)}</span>` +
      `<span class="ep">${esc(ep)}</span>`;
    item.addEventListener("click", () => chooseModel(llm.name));
    frag.appendChild(item);
  });
  if (!frag.childNodes.length) {
    const empty = document.createElement("div");
    empty.className = "load-empty";
    empty.textContent = "No models match";
    frag.appendChild(empty);
  }
  list.appendChild(frag);
}
function chooseModel(name) {
  if (busy) return;
  selected = name;
  localStorage.setItem("llmrunner-model", selected);
  closeLoadModal();
  pollNow();
  llmAction(name, "start");
}
function setLoadEnabled() {
  $("#btnLoadModel").disabled = busy;
}
function renderHero(llm) {
  const st = heroState(llm);
  $("#selName").textContent = llm.status === "stopped" ? "No model" : llm.name;
  $("#selName").title = llm.name;
  $("#selDot").classList.toggle("on", llm.status === "running");
  const pill = $("#statusPill");
  pill.className = "status " + st[0];
  $("#statusText").textContent = st[1];
  const running = llm.status !== "stopped";
  $("#btnLoadModel").hidden = running;
  $("#btnStop").hidden = !running;
  setLoadEnabled();
  renderThroughput(llm);
}
function renderThroughput(llm) {
  const hist = llm.history || [];
  const decode = llm.reachable ? llm.decode_tps : null;
  const prefill = llm.reachable ? llm.prefill_tps : null;
  $("#decodeVal").innerHTML = `${fmt(decode)}<small>tok/s</small>`;
  $("#prefillVal").innerHTML = `${fmt(prefill)}<small>tok/s</small>`;
  $("#decodeDelta").outerHTML = deltaHtml("decodeDelta", decode, hist, "decode_tps");
  $("#prefillDelta").outerHTML = deltaHtml("prefillDelta", prefill, hist, "prefill_tps");
  heroChart(hist);
}
function deltaHtml(id, cur, hist, key) {
  const vals = hist.filter((p) => Date.now() / 1000 - p.ts <= WINDOW_S && isNum(p[key])).map((p) => p[key]);
  let cls = "", txt = "";
  if (isNum(cur) && vals.length) {
    const d = cur - vals.reduce((a, b) => a + b, 0) / vals.length;
    cls = d >= 0 ? "up" : "down";
    txt = `${d >= 0 ? "▲" : "▼"} ${Math.abs(d).toFixed(1)} vs 5m avg`;
  }
  return `<div class="delta ${cls}" id="${id}">${txt}</div>`;
}
function heroChart(hist) {
  const svg = $("#heroSvg");
  svg.querySelectorAll("path").forEach((n) => n.remove());
  const t0 = Date.now() / 1000 - WINDOW_S;
  const mk = (key) => {
    const pts = hist.filter((p) => Date.now() / 1000 - p.ts <= WINDOW_S && isNum(p[key]));
    if (pts.length < 2) return null;
    const hi = Math.max(...pts.map((p) => p[key])) * 1.2 || 1;
    return pts.map((p) => [
      Math.max(Math.min(((p.ts - t0) / WINDOW_S) * 700, 700), 0),
      150 - (p[key] / hi) * 140,
    ]);
  };
  const add = (d, stroke, width, extra) => {
    const path = document.createElementNS(SVGNS, "path");
    path.setAttribute("d", d);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", stroke);
    path.setAttribute("stroke-width", String(width));
    if (extra) path.setAttribute("stroke-opacity", ".85");
    svg.appendChild(path);
  };
  const dec = mk("decode_tps");
  const pre = mk("prefill_tps");
  if (dec) {
    const area = document.createElementNS(SVGNS, "path");
    area.setAttribute("d", `M${dec.map((p) => `${p[0].toFixed(0)},${p[1].toFixed(0)}`).join(" L")} L700,150 L${dec[0][0].toFixed(0)},150 Z`);
    area.setAttribute("fill", "url(#gDec)");
    svg.appendChild(area);
    add(`M${dec.map((p) => `${p[0].toFixed(0)},${p[1].toFixed(0)}`).join(" L")}`, "var(--accent)", 2.2);
  }
  if (pre) add(`M${pre.map((p) => `${p[0].toFixed(0)},${p[1].toFixed(0)}`).join(" L")}`, "var(--warn)", 1.6, true);
}

/* ---------- hardware ---------- */
let gpuWarn = null;
function renderHw(hw) {
  updateCards("sys", hw);
  if (hw.gpu) {
    $("#rowGpu").hidden = false;
    if (gpuWarn) gpuWarn.hidden = true;
    updateCards("gpu", hw);
  } else {
    $("#rowGpu").hidden = true;
    if (!gpuWarn) {
      gpuWarn = document.createElement("div");
      gpuWarn.className = "warnbar";
      $("#rowGpu").after(gpuWarn);
    }
    gpuWarn.hidden = false;
    gpuWarn.textContent = "GPU metrics unavailable via NVML" +
      (hw.nvml_error ? ` (${hw.nvml_error})` : "") +
      ". Check the NVIDIA driver, or expose sensors through /sys/class/hwmon.";
  }
  updateCards("storage", hw);
}

/* ---------- start / stop ---------- */
async function llmAction(name, action) {
  busy = true;
  $("#btnLoadModel").disabled = true;
  if (action === "stop") $("#btnStop").classList.add("busy");
  try {
    const r = await jpost(`/api/llms/${encodeURIComponent(name)}/${action}`);
    toast(`${action} ${name}: ${r.ok ? "ok" : "failed"}${r.output ? "\n" + r.output.slice(-500) : ""}`, !r.ok);
  } catch (e) {
    toast(`${action} ${name}: ${e.message}`, true);
  }
  if (action === "stop") $("#btnStop").classList.remove("busy");
  busy = false;
  await pollNow();
  setTimeout(pollNow, 3000);
  setTimeout(pollNow, 8000);
}
$("#btnStop").addEventListener("click", () => llmAction(selected, "stop"));
function openLoadModal() {
  $("#loadModal").hidden = false;
  $("#loadSearch").value = "";
  renderLoadList();
  $("#loadSearch").focus();
}
function closeLoadModal() {
  $("#loadModal").hidden = true;
}
$("#btnLoadModel").addEventListener("click", openLoadModal);
$("#loadModalClose").addEventListener("click", closeLoadModal);
$("#loadModalBackdrop").addEventListener("click", closeLoadModal);
$("#loadSearch").addEventListener("input", () => renderLoadList($("#loadSearch").value));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeLoadModal();
});

/* ---------- main polling loop ---------- */
async function pollHw() {
  if (pollingHw) return;
  pollingHw = true;
  try {
    const hw = await jget("/api/hw");
    if (hw.ready) renderHw(hw);
    $("#livePill").classList.remove("off");
    $("#liveText").textContent = "live · 1s";
  } catch (e) {
    $("#livePill").classList.add("off");
    $("#liveText").textContent = "offline";
  } finally {
    pollingHw = false;
  }
}
async function pollLlms() {
  if (pollingLlms) return;
  pollingLlms = true;
  try {
    const lres = await jget("/api/llms");
    renderLlms(lres);
  } catch (e) {
    /* hw poll reports offline */
  } finally {
    pollingLlms = false;
  }
}
async function pollMetrics() {
  if (pollingMetrics) return;
  pollingMetrics = true;
  try {
    const res = await jget("/api/llms/metrics");
    const m = (res.llms || []).find((l) => l.name === selected);
    if (m) renderThroughput(m);
  } catch (e) {
    /* hw/status poll reports offline */
  } finally {
    pollingMetrics = false;
  }
}
async function pollNow() {
  await Promise.all([pollHw(), pollLlms(), pollMetrics()]);
}

buildCards("sys", CARDS.sys);
buildCards("gpu", CARDS.gpu);
buildCards("storage", CARDS.storage);
pollNow();
setInterval(pollHw, 1000);
setInterval(pollLlms, 5000);
setInterval(pollMetrics, 1000);
