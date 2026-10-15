'use strict';
/* DOM layer. All sizing math lives in calculator.js (window.LLMSizing). */
const { calculate, PRESETS, GPU_PRESETS } = window.LLMSizing;

const $ = (id) => document.getElementById(id);
const palette = ['#618cfa', '#56c9a8', '#f4b36a', '#b18bea', '#eb7c90', '#73c4d8'];
const NUMBER_IDS = ['params', 'activeParams', 'layers', 'hidden', 'kvHeads', 'headDim', 'latent', 'rotary', 'sequence', 'batch', 'users', 'prompt', 'loraRank', 'vram', 'bandwidth', 'tflops', 'gpus', 'tp', 'pp', 'reserve', 'metadata', 'buffers', 'bwEff', 'computeEff', 'layerOverheadUs', 'gpusPerNode', 'interconnectGBs', 'interNodeGBs', 'trainingMfu', 'trainingBatchSaturation'];
const SELECT_IDS = ['attention', 'precision', 'kvPrecision', 'ftMethod', 'optimizer', 'zero', 'runtime', 'model', 'gpu'];
const CHECK_IDS = ['checkpoint', 'flash'];

let mode = 'inference';
let lastResult = null;

function read() {
  const x = {};
  for (const id of NUMBER_IDS) x[id] = Number($(id).value);
  for (const id of SELECT_IDS) x[id] = $(id).value;
  for (const id of CHECK_IDS) x[id] = $(id).checked;
  return x;
}

function setMode(next) {
  mode = next;
  $('inferenceTab').classList.toggle('active', mode === 'inference');
  $('trainingTab').classList.toggle('active', mode === 'training');
  for (const id of ['ftMethodField', 'loraRankField', 'optimizerField', 'zeroField', 'trainingFlags']) $(id).classList.toggle('hidden', mode !== 'training');
  for (const id of ['kvPrecisionField', 'usersField', 'promptField']) $(id).classList.toggle('hidden', mode !== 'inference');
  update();
}

function applyPreset(table, key) {
  const preset = table[key];
  if (preset) for (const [k, v] of Object.entries(preset)) if ($(k)) $(k).value = v;
  update();
}
const applyModel = () => applyPreset(PRESETS, $('model').value);
const applyGPU = () => applyPreset(GPU_PRESETS, $('gpu').value);

function fixed(n, digits = 2) {
  return Number.isFinite(n) ? n.toLocaleString(undefined, { maximumFractionDigits: digits }) : '—';
}

function meterColor(pct) {
  if (pct > 90) return '#ed837e';
  if (pct > 75) return '#f3ba61';
  return '#56c9a8';
}

// Build rows with textContent so labels and messages are never parsed as HTML.
function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  Object.assign(node, props);
  node.append(...children);
  return node;
}

function renderBreakdown(r) {
  const entries = Object.entries(r.components_gib);
  $('bar').replaceChildren(
    ...entries.map(([k, v], i) => {
      const seg = el('span', { title: k });
      seg.style.width = (v / r.required_gib_per_gpu) * 100 + '%';
      seg.style.background = palette[i % palette.length];
      return seg;
    })
  );
  $('details').replaceChildren(
    ...entries.map(([k, v], i) => {
      const chip = el('i', { className: 'chip' });
      chip.style.background = palette[i % palette.length];
      return el('div', { className: 'row' }, el('span', {}, chip, k), el('b', { textContent: fixed(v) + ' GiB' }));
    })
  );
}

function render(r, x) {
  const training = r.mode === 'training';
  const perf = r.performance;
  $('required').textContent = fixed(r.required_gib_per_gpu) + ' GiB';
  $('available').textContent = fixed(r.usable_gib_per_gpu) + ' GiB';
  $('pct').textContent = fixed(r.pct, 1) + '%';
  $('status').textContent = r.status.replace('_', ' ').toUpperCase();
  $('status').className = r.pct > 90 ? 'bad' : 'good';
  $('meter').style.width = Math.min(100, r.pct) + '%';
  $('meter').style.background = meterColor(r.pct);
  $('speedTitle').textContent = training ? 'Training throughput' : 'Aggregate decode TPS';
  $('latencyTitle').textContent = training ? 'Optimizer steps / sec' : 'TTFT estimate';
  $('userTitle').textContent = training ? 'Local batch / GPU' : 'TPS / active stream';
  $('speed').textContent = fixed(training ? perf.training_tps : perf.aggregate_tps, 1);
  $('ttft').textContent = training ? fixed(perf.steps_per_second, 3) : fixed(perf.ttft_ms, 0) + ' ms';
  $('userSpeed').textContent = training ? fixed(r.localBatch, 0) : fixed(perf.per_stream_tps, 1);
  $('summary').textContent = `${r.replicas} data-parallel replica(s), ${x.tp} TP × ${x.pp} PP, local batch ${r.localBatch}. Status is based on usable VRAM after system reserve.`;
  renderBreakdown(r);
  $('warnings').replaceChildren(...r.warnings.map((s) => el('div', { className: 'warning', textContent: s })));
  const m = r.methodology;
  $('method').textContent = `Model weights: ${m.weights}. Attention cache: ${m.kv}. Performance: ${m.performance}. Attention activation/workspace and framework overhead are heuristic proxies, not exact traces.`;
}

function update() {
  try {
    const x = read();
    const r = calculate(x, mode);
    lastResult = r;
    render(r, x);
  } catch (e) {
    lastResult = null;
    $('status').textContent = 'INVALID INPUT';
    $('status').className = 'bad';
    $('warnings').replaceChildren(el('div', { className: 'warning', textContent: String(e.message) }));
  }
}

function saveJSON() {
  if (!lastResult) return;
  const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = el('a', { href: url, download: 'llm-sizing-estimate.json' });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 500);
}

function copyShare() {
  const url = new URL(location.href);
  url.searchParams.set('config', btoa(JSON.stringify({ mode, inputs: read() })));
  const done = () => alert('Configuration URL copied.');
  const fail = () => alert('Copy not available in this browser; use Export JSON.');
  if (!navigator.clipboard) return fail();
  navigator.clipboard.writeText(url.toString()).then(done).catch(fail);
}

function loadSharedConfig() {
  try {
    const q = new URL(location.href).searchParams.get('config');
    if (!q) return;
    const parsed = JSON.parse(atob(q));
    for (const [k, v] of Object.entries(parsed.inputs || {})) {
      if (CHECK_IDS.includes(k)) $(k).checked = Boolean(v);
      else if (NUMBER_IDS.includes(k) || SELECT_IDS.includes(k)) $(k).value = v;
    }
    if (parsed.mode === 'inference' || parsed.mode === 'training') mode = parsed.mode;
  } catch (e) {
    /* ignore malformed share links */
  }
}

for (const node of document.querySelectorAll('input,select')) {
  node.addEventListener('input', update);
  node.addEventListener('change', update);
}
loadSharedConfig();
setMode(mode);
