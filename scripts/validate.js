'use strict';
// Compares calculator.js against readings taken from the live ApX calculator.
// Usage: node scripts/validate.js [--markdown]
const fs = require('node:fs');
const path = require('node:path');
const { calculate, GPU_PRESETS } = require('../calculator.js');

const data = JSON.parse(fs.readFileSync(path.join(__dirname, '../data/apx-readings.json'), 'utf8'));

const DEFAULTS = {
  latent: 512, rotary: 64, sequence: 1024, batch: 1, users: 1, prompt: 1024, loraRank: 16,
  gpus: 1, tp: 1, pp: 1, reserve: 0, metadata: 0, buffers: 0.3, computeEff: 0.75,
  precision: 'fp16', kvPrecision: 'fp16', ftMethod: 'full', optimizer: 'adamw', zero: '0',
  runtime: 'generic', checkpoint: false, flash: true, gpusPerNode: 8, interconnectGBs: 64, interNodeGBs: 12.5,
};

function evaluate(row) {
  const gpu = data.gpus[row.gpu];
  const { preset, ...specs } = gpu;
  const x = { ...DEFAULTS, ...data.models[row.model], ...GPU_PRESETS[preset], ...specs, ...(row.precision ? { precision: row.precision } : {}), ...row.cfg };
  if (row.cfg && row.cfg.prompt === undefined && row.cfg.sequence) x.prompt = row.cfg.sequence;
  return calculate(x, row.mode || 'inference');
}

const pct = (ours, apx) => (ours / apx - 1) * 100;
const results = [];
for (const row of data.rows) {
  const r = evaluate(row);
  const out = { ...row, ours: {} };
  if (row.mode === 'training') {
    out.ours.gib = r.required_gib_per_gpu;
    out.ours.train_tps = r.performance.training_tps;
  } else {
    out.ours.tps = r.performance.per_stream_tps;
    out.ours.ttft = r.performance.ttft_ms;
    out.ours.gib = r.required_gib_per_gpu;
  }
  results.push(out);
}

const metrics = { tps: 'Decode tok/s (per user)', ttft: 'TTFT', gib: 'VRAM per GPU', train_tps: 'Training tok/s' };
const stats = {};
for (const r of results) {
  for (const key of Object.keys(metrics)) {
    const apx = key === 'train_tps' ? r.train_tps : r[key];
    if (apx === undefined || r.ours[key] === undefined) continue;
    (stats[key] ||= []).push(Math.abs(pct(r.ours[key], apx)));
  }
}
const median = (a) => { const s = [...a].sort((p, q) => p - q); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const quantile = (a, q) => [...a].sort((p, q2) => p - q2)[Math.min(a.length - 1, Math.floor(q * a.length))];

const fmt = (v, d = 0) => (v >= 100 ? v.toFixed(0) : v.toFixed(d));
const cell = (apx, ours, d = 0) => (apx === undefined ? '—' : `${fmt(apx, d)} → ${fmt(ours, d)} (${pct(ours, apx) >= 0 ? '+' : ''}${pct(ours, apx).toFixed(0)}%)`);

if (process.argv.includes('--markdown')) {
  console.log('| Metric | Readings | Median abs. error | 90th percentile |');
  console.log('|---|---:|---:|---:|');
  for (const [key, label] of Object.entries(metrics)) {
    const a = stats[key]; if (!a) continue;
    console.log(`| ${label} | ${a.length} | ${median(a).toFixed(1)}% | ${quantile(a, 0.9).toFixed(1)}% |`);
  }
  let group = null;
  for (const r of results) {
    if (r.group !== group) {
      group = r.group;
      console.log(`\n**${group}**\n`);
      console.log(r.mode === 'training'
        ? '| Case | VRAM GiB (ApX → ours) | Training tok/s (ApX → ours) |\n|---|---|---|'
        : '| Case | Decode tok/s (ApX → ours) | TTFT ms (ApX → ours) | VRAM GiB/GPU (ApX → ours) |\n|---|---|---|---|');
    }
    if (r.mode === 'training') console.log(`| ${r.label} | ${cell(r.gib, r.ours.gib, 1)} | ${cell(r.train_tps, r.ours.train_tps)} |`);
    else console.log(`| ${r.label} | ${cell(r.tps, r.ours.tps)} | ${cell(r.ttft, r.ours.ttft)} | ${cell(r.gib, r.ours.gib, 1)} |`);
  }
} else {
  for (const [key, label] of Object.entries(metrics)) {
    const a = stats[key]; if (!a) continue;
    console.log(`${label.padEnd(26)} n=${String(a.length).padStart(2)}  median |err| ${median(a).toFixed(1)}%  p90 ${quantile(a, 0.9).toFixed(1)}%  max ${Math.max(...a).toFixed(1)}%`);
  }
}
