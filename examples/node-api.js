'use strict';
// Size Llama 3.1 70B (INT4) on 4x A100 with tensor parallelism.
// Run: node examples/node-api.js
const { calculate, PRESETS, GPU_PRESETS } = require('../calculator.js');

const result = calculate({
  ...PRESETS.llama70, // 70B, GQA, 80 layers
  ...GPU_PRESETS.a100, // 80 GB, 2039 GB/s, 312 TFLOPS + fitted efficiency profile
  precision: 'int4',
  kvPrecision: 'fp16',
  sequence: 8192,
  prompt: 1024,
  batch: 1,
  users: 1,
  gpus: 4,
  tp: 4,
  pp: 1,
  gpusPerNode: 8,
  interconnectGBs: 64, // PCIe 4.0 x16
  interNodeGBs: 12.5, // 100 GbE
  reserve: 1.5,
  metadata: 0,
  runtime: 'vllm',
  computeEff: 0.75,
  // training-only fields are ignored in inference mode
  loraRank: 16,
  ftMethod: 'full',
  optimizer: 'adamw',
  zero: '0',
  checkpoint: false,
  flash: true,
  latent: 512,
  rotary: 64,
});

console.log(`status           ${result.status}`);
console.log(`VRAM per GPU     ${result.required_gib_per_gpu.toFixed(1)} GiB of ${result.usable_gib_per_gpu.toFixed(1)} usable`);
console.log(`decode (1 user)  ${result.performance.per_stream_tps.toFixed(0)} tok/s`);
console.log(`TTFT             ${result.performance.ttft_ms.toFixed(0)} ms`);
console.table(result.components_gib);
