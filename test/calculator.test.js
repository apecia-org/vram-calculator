'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { calculate, classify, PRESETS, GPU_PRESETS } = require('../calculator.js');

const GiB = 1024 ** 3;

// Mirrors the defaults in index.html (Llama 3 8B on one RTX 4090).
function config(overrides = {}) {
  return {
    ...PRESETS.llama8,
    latent: 512,
    rotary: 64,
    ...GPU_PRESETS['4090'],
    sequence: 4096,
    batch: 1,
    users: 1,
    prompt: 1024,
    loraRank: 16,
    gpus: 1,
    tp: 1,
    pp: 1,
    reserve: 1.5,
    metadata: 0,
    buffers: 0.3,
    computeEff: 0.75,
    gpusPerNode: 8,
    interconnectGBs: 64,
    interNodeGBs: 12.5,
    precision: 'fp16',
    kvPrecision: 'fp16',
    ftMethod: 'full',
    optimizer: 'adamw',
    zero: '0',
    runtime: 'vllm',
    checkpoint: false,
    flash: true,
    ...overrides,
  };
}

const near = (actual, expected, tol = 1e-6) => assert.ok(Math.abs(actual - expected) <= tol * Math.max(1, Math.abs(expected)), `${actual} !~ ${expected}`);
const run = (o, m = 'inference') => calculate(config(o), m);

test('weights: Llama 3 8B FP16 is params × 2 bytes / 2^30', () => {
  near(run().components_gib['Model weights'], (8.03e9 * 2) / GiB);
});

test('weights: quantization metadata overhead is applied', () => {
  near(run({ precision: 'int4' }).components_gib['Model weights'], ((8.03e9 * 0.5) / GiB) * 1.04);
  near(run({ precision: 'fp16', metadata: 10 }).components_gib['Model weights'], ((8.03e9 * 2) / GiB) * 1.1);
});

test('KV cache: Llama 3 8B GQA at 8192 tokens, batch 1 is exactly 1 GiB', () => {
  near(run({ sequence: 8192 }).components_gib['KV cache'], 1);
});

test('KV cache: Llama 3 70B GQA at 8192 tokens is 2.5 GiB', () => {
  near(run({ ...PRESETS.llama70, sequence: 8192 }).components_gib['KV cache'], 2.5);
});

test('KV cache: MLA uses layers × (latent + rotary), not 2 × heads × head_dim', () => {
  const r = run({ ...PRESETS.deepseek, sequence: 1000, kvPrecision: 'fp16' });
  near(r.components_gib['KV cache'], (61 * (512 + 64) * 1000 * 2) / GiB);
});

test('KV cache: FP8 cache is half of FP16', () => {
  const fp16 = run({ sequence: 8192 }).components_gib['KV cache'];
  near(run({ sequence: 8192, kvPrecision: 'fp8' }).components_gib['KV cache'], fp16 / 2);
});

test('parallelism: TP=2 halves per-GPU weights and KV', () => {
  const one = run({ sequence: 8192 });
  const two = run({ sequence: 8192, gpus: 2, tp: 2 });
  near(two.components_gib['Model weights'], one.components_gib['Model weights'] / 2);
  near(two.components_gib['KV cache'], one.components_gib['KV cache'] / 2);
  assert.equal(two.replicas, 1);
});

test('parallelism: extra GPUs beyond TP×PP become data-parallel replicas', () => {
  const r = run({ gpus: 8, tp: 2, pp: 1, batch: 8 });
  assert.equal(r.replicas, 4);
  assert.equal(r.localBatch, 2);
});

test('memory is monotonic in batch, sequence and parameter count', () => {
  const base = run().required_gib_per_gpu;
  assert.ok(run({ batch: 4 }).required_gib_per_gpu > base);
  assert.ok(run({ sequence: 16384 }).required_gib_per_gpu > base);
  assert.ok(run({ params: 20 }).required_gib_per_gpu > base);
});

test('decode step: memory + overhead + per-sequence compute', () => {
  const r = run({ sequence: 1, prompt: 1 });
  const readGB = ((8.03e9 * 2) / GiB + r.components_gib['KV cache']) * (GiB / 1e9);
  const expected = (readGB / (1008 * 0.69)) * 1000 + (32 * 84) / 1000 + ((2 * 8.03e9) / (165e12 * 0.215)) * 1000;
  near(r.performance.step_ms, expected);
  near(r.performance.aggregate_tps, 1000 / expected);
  assert.equal(r.performance.bound, 'memory');
});

test('decode: bytes read are not divided by replica count', () => {
  const single = run({ sequence: 1, prompt: 1 }).performance.aggregate_tps;
  const dual = run({ sequence: 1, prompt: 1, gpus: 2, batch: 2 }).performance.aggregate_tps;
  near(dual, single * 2);
});

test('decode: per-user speed divides the aggregate by max(users, batch)', () => {
  const one = run({ sequence: 1024 }).performance;
  const queued = run({ sequence: 1024, users: 8 }).performance;
  near(queued.aggregate_tps, one.aggregate_tps);
  near(queued.per_stream_tps, one.aggregate_tps / 8);
  const batched = run({ sequence: 1024, batch: 8, users: 1 }).performance;
  near(batched.per_stream_tps, batched.aggregate_tps / 8);
});

test('decode: batch scaling is sublinear and saturates at the compute ceiling', () => {
  const tps = (batch) => run({ sequence: 1024, precision: 'int4', batch, users: batch }).performance.aggregate_tps;
  const ratio = tps(8) / tps(1);
  assert.ok(ratio > 4 && ratio < 6, `ratio ${ratio}`);
  const ceiling = (165e12 * 0.215) / (2 * 8.03e9);
  assert.ok(tps(8) < ceiling && tps(1) < tps(2) && tps(2) < tps(4));
});

test('TTFT does not change with concurrent users', () => {
  near(run({ users: 8 }).performance.ttft_ms, run({ users: 1 }).performance.ttft_ms);
});

test('decode: higher bandwidth never lowers throughput', () => {
  const slow = run({ bandwidth: 1000 }).performance.aggregate_tps;
  const fast = run({ bandwidth: 2000 }).performance.aggregate_tps;
  assert.ok(fast >= slow);
});

// Per-user tok/s read from apxml.com/tools/vram-calculator (v4.2):
// Llama 3.1 8B, 1,024 tokens, FP16 KV cache, batch = users.
test('regression vs ApX live calculator (decode)', () => {
  const gpus = {
    rtx4090: GPU_PRESETS['4090'],
    h100pcie: { ...GPU_PRESETS.h100, bandwidth: 2000, tflops: 756 },
    h100: { ...GPU_PRESETS.h100, tflops: 989.5 },
    a100: { ...GPU_PRESETS.a100, vram: 40, bandwidth: 1555 },
  };
  const rows = [
    ['rtx4090', 'fp16', 1, 38], ['rtx4090', 'int4', 1, 111], ['rtx4090', 'int4', 8, 67],
    ['h100pcie', 'fp16', 1, 99], ['h100pcie', 'fp8', 1, 179], ['h100pcie', 'int8', 1, 179], ['h100pcie', 'int4', 1, 283],
    ['h100', 'fp16', 1, 156], ['h100', 'int8', 1, 268], ['h100', 'int4', 1, 398],
    ['a100', 'fp16', 1, 79], ['a100', 'int8', 1, 145], ['a100', 'int4', 1, 235],
    ['a100', 'fp16', 8, 69], ['a100', 'int8', 8, 114], ['a100', 'int4', 8, 153],
  ];
  for (const [gpu, precision, batch, apx] of rows) {
    const p = run({ ...gpus[gpu], precision, batch, users: batch, sequence: 1024, runtime: 'generic', reserve: 0 }).performance;
    assert.ok(Math.abs(p.per_stream_tps / apx - 1) <= 0.12, `${gpu} ${precision} b${batch}: ${p.per_stream_tps} vs ${apx}`);
  }
});

test('regression vs ApX live calculator (long context, 8B FP16, A100 40GB)', () => {
  const a100 = { ...GPU_PRESETS.a100, vram: 40, bandwidth: 1555 };
  const rows = [[8192, 75, 0.565], [32768, 64, 2.97], [65536, 53, 7.99]];
  for (const [n, tps, ttft] of rows) {
    const p = run({ ...a100, sequence: n, prompt: n, runtime: 'generic', reserve: 0 }).performance;
    assert.ok(Math.abs(p.per_stream_tps / tps - 1) <= 0.1, `${n}: tps ${p.per_stream_tps} vs ${tps}`);
    assert.ok(Math.abs(p.ttft_ms / 1000 / ttft - 1) <= 0.15, `${n}: ttft ${p.ttft_ms} vs ${ttft}`);
  }
});

test('regression vs ApX live calculator (Llama 3.1 70B INT4, A100 40GB, PCIe)', () => {
  const a100 = { ...GPU_PRESETS.a100, vram: 40, bandwidth: 1555 };
  const rows = [
    [{}, 31, 671], [{ gpus: 2, tp: 2 }, 62, 339], [{ gpus: 4, tp: 4 }, 96, 208], [{ gpus: 4, pp: 4 }, 34, 575],
  ];
  for (const [par, tps, ttft] of rows) {
    const p = run({ ...PRESETS.llama70, ...a100, ...par, precision: 'int4', sequence: 1024, runtime: 'generic', reserve: 0 }).performance;
    assert.ok(Math.abs(p.per_stream_tps / tps - 1) <= 0.12, `${JSON.stringify(par)}: tps ${p.per_stream_tps} vs ${tps}`);
    assert.ok(Math.abs(p.ttft_ms / ttft - 1) <= 0.2, `${JSON.stringify(par)}: ttft ${p.ttft_ms} vs ${ttft}`);
  }
});

test('regression vs ApX live calculator (70B INT4 per-GPU VRAM and speed, A100 40GB)', () => {
  const a100 = { ...GPU_PRESETS.a100, vram: 40, bandwidth: 1555 };
  const rows = [
    [{}, 36.6, 31], [{ gpus: 2, tp: 2 }, 21.0, 62], [{ gpus: 4, tp: 4 }, 12.5, 96], [{ gpus: 8, tp: 8 }, 8.2, 138],
    [{ gpus: 4, pp: 4 }, 12.9, 34], [{ gpus: 8, tp: 2, pp: 4 }, 8.5, 60],
  ];
  for (const [par, gib, tps] of rows) {
    const r = run({ ...PRESETS.llama70, ...a100, ...par, precision: 'int4', sequence: 1024, runtime: 'generic', reserve: 0 });
    assert.ok(Math.abs(r.required_gib_per_gpu / gib - 1) <= 0.1, `${JSON.stringify(par)}: ${r.required_gib_per_gpu} GiB vs ${gib}`);
    assert.ok(Math.abs(r.performance.per_stream_tps / tps - 1) <= 0.12, `${JSON.stringify(par)}: ${r.performance.per_stream_tps} tok/s vs ${tps}`);
  }
});

// A100 40GB, TP8 (8B INT4 and 70B INT4), 1,024 tokens. Values from apxml.com/tools/vram-calculator.
test('regression vs ApX live calculator (interconnect)', () => {
  const a100 = { ...GPU_PRESETS.a100, vram: 40, bandwidth: 1555, precision: 'int4', sequence: 1024, runtime: 'generic', reserve: 0, gpus: 8, tp: 8 };
  const rows = [
    // [model, interconnect GB/s, tok/s, TTFT ms]
    [PRESETS.llama8, 64, 579, 27], [PRESETS.llama8, 32, 534, 36], [PRESETS.llama8, 12.5, 383, 50], [PRESETS.llama8, 1.25, 17, 315],
    [PRESETS.llama70, 64, 138, 140], [PRESETS.llama70, 12.5, 68, 240], [PRESETS.llama70, 1.25, 2, 1700],
  ];
  for (const [model, bw, tps, ttft] of rows) {
    const p = run({ ...model, ...a100, interconnectGBs: bw }).performance;
    assert.ok(Math.abs(p.per_stream_tps / tps - 1) <= 0.25, `${model.params}B @${bw}: ${p.per_stream_tps} tok/s vs ${tps}`);
    assert.ok(Math.abs(p.ttft_ms / ttft - 1) <= 0.3, `${model.params}B @${bw}: ${p.ttft_ms} ms vs ${ttft}`);
  }
});

test('interconnect: slower links never speed things up; multi-node sits between intra and inter bandwidth', () => {
  const a100 = { ...GPU_PRESETS.a100, vram: 40, bandwidth: 1555, ...PRESETS.llama70, precision: 'int4', gpus: 8, tp: 8 };
  const fast = run({ ...a100, interconnectGBs: 64 }).performance.per_stream_tps;
  const slow = run({ ...a100, interconnectGBs: 12.5 }).performance.per_stream_tps;
  const split = run({ ...a100, gpusPerNode: 4 }).performance.per_stream_tps; // 2 nodes, 64 GB/s inside, 12.5 between
  assert.ok(slow < split && split < fast, `${slow} < ${split} < ${fast}`);
  // ApX: 2 nodes x 4 A100, TP8, PCIe4 + 100GbE = 101 tok/s
  assert.ok(Math.abs(split / 101 - 1) <= 0.2, `split ${split}`);
});

test('TTFT: batch prompts sharing one user prefill together (ApX: ~6x at batch 8)', () => {
  const single = run({ ...GPU_PRESETS['4090'], precision: 'int4', sequence: 1024, batch: 8, users: 8 }).performance.ttft_ms;
  const queued = run({ ...GPU_PRESETS['4090'], precision: 'int4', sequence: 1024, batch: 8, users: 1 }).performance.ttft_ms;
  assert.ok(queued / single > 4.5 && queued / single < 7.5, `ratio ${queued / single}`);
});

test('MoE: weights touched grow with batch (Mixtral-like, ApX-style 7B active)', () => {
  const moe = { params: 46.7, activeParams: 7, layers: 32, hidden: 4096, kvHeads: 8, headDim: 128, precision: 'int4', ...GPU_PRESETS.a100, vram: 40, bandwidth: 1555, sequence: 1024, runtime: 'generic', reserve: 0 };
  const b1 = run({ ...moe }).performance.per_stream_tps;
  const b8 = run({ ...moe, batch: 8, users: 8 }).performance.per_stream_tps;
  assert.ok(Math.abs(b1 / 259 - 1) <= 0.2, `b1 ${b1}`);
  assert.ok(Math.abs(b8 / 70 - 1) <= 0.25, `b8 ${b8}`);
  const dense = run({ ...moe, activeParams: 46.7, batch: 8, users: 8 }).performance.per_stream_tps;
  assert.ok(dense < b8);
});

test('parallelism: TP speeds up a single stream, PP barely does', () => {
  const one = run({ ...PRESETS.llama70, precision: 'int4' }).performance.per_stream_tps;
  const tp4 = run({ ...PRESETS.llama70, precision: 'int4', gpus: 4, tp: 4 }).performance.per_stream_tps;
  const pp4 = run({ ...PRESETS.llama70, precision: 'int4', gpus: 4, pp: 4 }).performance.per_stream_tps;
  assert.ok(tp4 / one > 2.5);
  assert.ok(pp4 / one < 1.3);
});

test('prefill attention grows quadratically with prompt length', () => {
  const t = (n) => run({ sequence: n, prompt: n }).performance.ttft_ms;
  assert.ok(t(65536) / t(32768) > 2.1);
});

test('regression vs ApX live calculator (TTFT, batch 1)', () => {
  const rows = [
    [GPU_PRESETS['4090'], 'fp16', 168], [GPU_PRESETS['4090'], 'int4', 150],
    [{ ...GPU_PRESETS.a100, vram: 40, bandwidth: 1555 }, 'fp16', 81], [{ ...GPU_PRESETS.a100, vram: 40, bandwidth: 1555 }, 'int4', 72],
    [{ ...GPU_PRESETS.h100, tflops: 989.5 }, 'int4', 29],
  ];
  for (const [gpu, precision, apx] of rows) {
    const p = run({ ...gpu, precision, sequence: 1024 }).performance;
    assert.ok(Math.abs(p.ttft_ms / apx - 1) <= 0.2, `${precision}: ${p.ttft_ms} vs ${apx}`);
  }
});

test('training: ZeRO shards optimizer states, gradients and weights across DP replicas only', () => {
  const dp = { gpus: 8, tp: 2, pp: 1, batch: 8, vram: 80 };
  const z0 = run({ ...dp, zero: '0' }, 'training').components_gib;
  const z1 = run({ ...dp, zero: '1' }, 'training').components_gib;
  const z2 = run({ ...dp, zero: '2' }, 'training').components_gib;
  const z3 = run({ ...dp, zero: '3' }, 'training').components_gib;
  const replicas = 4;
  near(z1['Optimizer states'], z0['Optimizer states'] / replicas);
  near(z1.Gradients, z0.Gradients);
  near(z2.Gradients, z0.Gradients / replicas);
  near(z2['Model weights'], z0['Model weights']);
  near(z3['Model weights'], z0['Model weights'] / replicas);
});

test('training: full fine-tune optimizer is 12 bytes/param for AdamW, 4 for SGD', () => {
  const p = (8.03e9) / GiB;
  near(run({}, 'training').components_gib['Optimizer states'], p * 12);
  near(run({ optimizer: 'sgd' }, 'training').components_gib['Optimizer states'], p * 4);
});

test('training: gradient bytes follow weight precision for FP32 full fine-tune', () => {
  const p = 8.03e9 / GiB;
  near(run({ precision: 'fp16' }, 'training').components_gib.Gradients, p * 2);
  near(run({ precision: 'fp32' }, 'training').components_gib.Gradients, p * 4);
});

test('training: QLoRA uses 4.35-bit base weights and a far smaller trainable set', () => {
  const q = run({ ftMethod: 'qlora' }, 'training').components_gib;
  near(q['Model weights'], ((8.03e9 * 4.35) / 8 / GiB) * 1.01);
  assert.ok(q['Optimizer states'] < 1);
});

test('training: checkpointing and flash attention reduce saved activations', () => {
  const base = run({ flash: false }, 'training').components_gib['Saved activations'];
  assert.ok(run({ flash: true }, 'training').components_gib['Saved activations'] < base);
  near(run({ flash: false, checkpoint: true }, 'training').components_gib['Saved activations'], base * 0.4);
});

// Llama 3.1 8B fine-tuning on one H100 80GB, 1,024 tokens (apxml.com/tools/vram-calculator, calibrated).
test('regression vs ApX live calculator (fine-tuning)', () => {
  const h100 = { ...GPU_PRESETS.h100, tflops: 989.5, reserve: 0, runtime: 'hf' };
  const rows = [
    [{ ftMethod: 'full' }, 128.5, null],
    [{ ftMethod: 'lora' }, 24.2, 5729],
    [{ ftMethod: 'lora', batch: 8 }, 66.4, 8188],
    [{ ftMethod: 'qlora' }, 13.5, 5729],
    [{ ftMethod: 'qlora', batch: 8 }, 55.6, 8188],
  ];
  for (const [o, gb, tps] of rows) {
    const r = run({ ...h100, sequence: 1024, ...o }, 'training');
    assert.ok(Math.abs(r.required_gib_per_gpu / gb - 1) <= 0.08, `${JSON.stringify(o)}: ${r.required_gib_per_gpu} vs ${gb}`);
    if (tps) assert.ok(Math.abs(r.performance.training_tps / tps - 1) <= 0.05, `${JSON.stringify(o)}: ${r.performance.training_tps} vs ${tps}`);
  }
});

test('training: ZeRO-3 without replicas warns', () => {
  assert.ok(run({ zero: '3' }, 'training').warnings.some((w) => w.includes('ZeRO-3')));
});

test('status thresholds', () => {
  assert.equal(classify(50), 'okay');
  assert.equal(classify(50.1), 'moderate');
  assert.equal(classify(75.1), 'high');
  assert.equal(classify(90.1), 'very_high');
  assert.equal(classify(100.1), 'insufficient');
});

test('status: 70B FP16 on one 4090 is insufficient and warns', () => {
  const r = run(PRESETS.llama70);
  assert.equal(r.status, 'insufficient');
  assert.ok(r.warnings.some((w) => w.includes('exceeds usable GPU memory')));
});

test('validation rejects impossible inputs', () => {
  assert.throws(() => run({ gpus: 3, tp: 2 }), /divide GPU count/);
  assert.throws(() => run({ users: 0 }), /users must be positive/);
  assert.throws(() => run({ precision: 'fp3' }), /unsupported value/);
  assert.throws(() => run({ bwEff: 1.5 }), /bwEff/);
  assert.throws(() => run({ tp: 1.5 }), /whole number/);
  assert.throws(() => run({ batch: NaN }), /batch/);
});

test('validation: training ignores inference-only fields', () => {
  assert.doesNotThrow(() => run({ users: 0, prompt: 0, kvPrecision: 'none' }, 'training'));
});
