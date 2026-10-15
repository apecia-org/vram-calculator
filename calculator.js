'use strict';
/*
 * Pure sizing logic: no DOM access, so it runs in the browser (window.LLMSizing)
 * and in Node (require('./calculator.js')).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.LLMSizing = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  const GiB = 1024 ** 3;
  const GB = 1e9;

  const PRESETS = {
    llama8: { params: 8.03, activeParams: 8.03, layers: 32, hidden: 4096, attention: 'gqa', kvHeads: 8, headDim: 128 },
    llama70: { params: 70.55, activeParams: 70.55, layers: 80, hidden: 8192, attention: 'gqa', kvHeads: 8, headDim: 128 },
    mistral7: { params: 7.24, activeParams: 7.24, layers: 32, hidden: 4096, attention: 'gqa', kvHeads: 8, headDim: 128 },
    deepseek: { params: 671, activeParams: 37, layers: 61, hidden: 7168, attention: 'mla', kvHeads: 128, headDim: 128, latent: 512, rotary: 64 },
  };

  // buffers: ApX reports ~1.7 GiB more framework memory on A100/H100 than on the 4090.
  // bwEff / layerOverheadUs are fitted to ApX readings (Llama 3.1 8B and 70B, decode):
  // GDDR consumer cards ~0.69 / 84 us per layer, HBM datacenter cards ~0.93 / 34 us per layer.
  const GPU_PRESETS = {
    4090: { vram: 24, bandwidth: 1008, tflops: 165, bwEff: 0.69, layerOverheadUs: 84, buffers: 0.3 },
    a100: { vram: 80, bandwidth: 2039, tflops: 312, bwEff: 0.93, layerOverheadUs: 34, buffers: 2.0 },
    h100: { vram: 80, bandwidth: 3350, tflops: 989, bwEff: 0.93, layerOverheadUs: 34, buffers: 2.0 },
    h200: { vram: 141, bandwidth: 4800, tflops: 989, bwEff: 0.93, layerOverheadUs: 34, buffers: 2.0 },
  };

  // bits per weight element and fractional metadata overhead (scales, zero points)
  const FORMATS = {
    fp32: { bits: 32, overhead: 0 },
    fp16: { bits: 16, overhead: 0 },
    fp8: { bits: 8, overhead: 0 },
    int8: { bits: 8, overhead: 0.01 },
    int4: { bits: 4, overhead: 0.04 }, // 4.156 bpw for block 128 + fp16 scale
    q4_k_m: { bits: 4.85, overhead: 0 },
    q5_k_m: { bits: 5.7, overhead: 0 },
    q8_0: { bits: 8.5, overhead: 0 },
  };

  // Every number below is a labeled heuristic, not a published ApX constant.
  const HEURISTICS = {
    runtimeBaseGiB: { vllm: 0.65, hf: 0.85, llamacpp: 0.35, generic: 0.5 },
    // ApX: per-GPU VRAM is ~0.8 GiB higher on a multi-GPU layout, on top of the ~1.7 GiB datacenter-card buffers preset.
    multiGpuOverheadGiB: 0.8,
    inferenceWorkspaceFloorGiB: 0.12,
    inferenceWorkspaceTokenCap: 4096,
    inferenceWorkspaceMultiplier: 4,
    // Fraction of peak FLOPs reached by the per-sequence decode math (fit to ApX batch 1 vs 8).
    decodeComputeEff: 0.215,
    // Prefill attention (2 * layers * hidden * N^2 causal FLOPs) runs near peak with fused kernels.
    prefillAttentionEff: 1.0,
    // ApX: batch 8 with 1 user takes ~5.75-7x the TTFT of a single prompt (8 prompts prefill together); batch = users does not.
    queuedPrefillExponent: 0.9,
    // ApX H100 reading: FP8 prefill runs on FP8 tensor cores (~1.4x faster TTFT than INT8).
    fp8PrefillSpeedup: 1.8,
    // ApX shows ~1.1x single-stream speedup from 4 pipeline stages: stages run sequentially.
    ppSpeedupPerStage: 0.03,
    // Interconnect loss per TP doubling beyond 2 GPUs (ApX, PCIe 4.0 x16, 70B: TP4 = 3.1x decode, 3.2x prefill).
    tpDecodeLossPerDoubling: 0.08,
    // Per-layer synchronization cost of tensor-parallel prefill on PCIe 4.0 (ApX 70B: +34 ms at TP4, +51 ms at TP8).
    tpPrefillSyncMsPerLayer: 0.85,
    // Interconnect model, fitted to ApX (A100 40GB, TP8; PCIe 4.0 / 100GbE / 10GbE; 8B and 70B).
    // The losses above are the PCIe 4.0 x16 (64 GB/s) baseline; only the difference is added.
    referenceInterconnectGBs: 64,
    decodeCommMsRef: 0.9, // 8B-class model (32 layers, hidden 4096), TP8, 12.5 GB/s
    decodeCommBwExponent: 1.8,
    decodeCommTpExponent: 1.1,
    decodeCommHiddenExponent: 1.75,
    prefillCommKappa: 0.36, // fraction of ring all-reduce time that is not hidden
    // Share of a TP group's all-reduce time spent on the inter-node link when TP spans nodes.
    interNodeDecodeShare: 0.5,
    interNodePrefillShare: 0.9,
    qloraBits: 4.35,
    qloraMetadata: 0.01,
    qloraExtraRuntimeGiB: 0.4,
    // ~42M trainable params for an 8B model at rank 16 on all linear layers (0.52%).
    loraTrainableFractionAtRank16: 0.0052,
    // trainer buffers beyond the runtime base and GPU preset buffers (ApX HF-trainer readings).
    trainingFrameworkGiB: 0.1,
    // Logits/loss buffers per token, assuming a ~128K vocabulary (fp32 logits + grad + bf16 copy).
    trainingLogitsGiBPerToken: 1.25 / 1024,
    // Fit to ApX calibrated H100 HF-trainer readings: 5,729 tok/s at batch 1, 8,188 at batch 8 (8B).
    trainingMfu: 0.42,
    trainingBatchSaturation: 0.52,
    lowPrecisionGradientBytes: 2,
    optimizerBytesPerParam: { adamw: 12, adam8: 6, sgd: 4 },
    activationFactor: { flash: 19, standard: 30 },
    checkpointActivationScale: 0.4,
    trainingFlopsPerParamToken: 6,
  };

  const POSITIVE_COMMON = ['interconnectGBs', 'interNodeGBs', 'gpusPerNode', 'params', 'activeParams', 'layers', 'hidden', 'kvHeads', 'headDim', 'sequence', 'batch', 'gpus', 'tp', 'pp', 'vram', 'bandwidth', 'tflops'];
  const INTEGERS = ['gpus', 'tp', 'pp', 'batch'];

  function bytes(bits) {
    return bits / 8;
  }

  function validate(x, m) {
    if (m !== 'inference' && m !== 'training') throw Error('mode must be inference or training');
    const positive = POSITIVE_COMMON.slice();
    if (m === 'inference') positive.push('users', 'prompt');
    if (m === 'training' && x.ftMethod !== 'full') positive.push('loraRank');
    if (x.attention === 'mla') positive.push('latent');
    for (const key of positive) if (!(x[key] > 0)) throw Error(key + ' must be positive');
    for (const key of INTEGERS) if (!Number.isInteger(x[key])) throw Error(key + ' must be a whole number');
    for (const key of ['reserve', 'buffers', 'metadata', 'rotary', 'layerOverheadUs', 'interconnectGBs', 'interNodeGBs']) if (!(x[key] >= 0)) throw Error(key + ' must not be negative');
    for (const key of ['bwEff', 'computeEff']) if (!(x[key] > 0 && x[key] <= 1)) throw Error(key + ' must be in (0, 1]');

    const choices = { attention: ['mha', 'gqa', 'mla'], precision: Object.keys(FORMATS), runtime: Object.keys(HEURISTICS.runtimeBaseGiB) };
    if (m === 'inference') choices.kvPrecision = Object.keys(FORMATS);
    else Object.assign(choices, { ftMethod: ['full', 'lora', 'qlora'], optimizer: Object.keys(HEURISTICS.optimizerBytesPerParam), zero: ['0', '1', '2', '3'] });
    for (const [key, allowed] of Object.entries(choices)) if (!allowed.includes(x[key])) throw Error(key + ' has an unsupported value: ' + x[key]);

    if (x.tp * x.pp > x.gpus || x.gpus % (x.tp * x.pp) !== 0) throw Error('Tensor × pipeline parallel degree must divide GPU count.');
  }

  function sum(components) {
    return Object.values(components).reduce((a, b) => a + b, 0);
  }

  // Effective all-reduce bandwidth (GB/s) for a TP group; spans nodes when tp > gpusPerNode.
  function commBandwidth(x, interNodeShare) {
    if (x.tp <= x.gpusPerNode) return x.interconnectGBs;
    return 1 / ((1 - interNodeShare) / x.interconnectGBs + interNodeShare / x.interNodeGBs);
  }

  function inferenceMemory(x, ctx) {
    const { localBatch, stages, weights, overhead } = ctx;
    const H = HEURISTICS;
    const kvBytes = bytes(FORMATS[x.kvPrecision].bits);
    const kvElementsPerToken = x.attention === 'mla' ? x.layers * (x.latent + x.rotary) : 2 * x.layers * x.kvHeads * x.headDim;
    const kvTotalGiB = (kvElementsPerToken * localBatch * x.sequence * kvBytes) / GiB; // one replica
    const kv = kvTotalGiB / stages;
    const tokens = Math.min(x.sequence, H.inferenceWorkspaceTokenCap);
    const activations = Math.max(H.inferenceWorkspaceFloorGiB, ((localBatch * x.hidden * tokens * 2) / GiB) * H.inferenceWorkspaceMultiplier / x.tp);
    const components = { 'Model weights': weights, 'KV cache': kv, 'Activations / workspace': activations, 'Runtime + communication': overhead };
    return { components, kvTotalGiB };
  }

  /*
   * One decode step for a replica:
   *   step = (weights + batch * KV) / (BW * bwEff)      memory traffic
   *        + layers * layerOverheadUs                   kernel launch / framework latency
   *        + batch * 2 * active / (TFLOPS * eps)        per-sequence math
   * Every sequence in the batch emits one token per step. Tensor parallelism splits
   * the work across devices; pipeline stages run sequentially (small gain only).
   */
  function inferencePerformance(x, ctx, kvTotalGiB) {
    const { replicas, localBatch, modelTotalGiB } = ctx;
    const H = HEURISTICS;
    const active = x.activeParams * 1e9;
    const ppGain = 1 + H.ppSpeedupPerStage * (x.pp - 1);
    const doublings = Math.max(0, Math.log2(x.tp) - 1);
    const speedup = x.tp * ppGain / (1 + H.tpDecodeLossPerDoubling * doublings);
    const prefillSpeedup = x.tp * ppGain;

    // MoE: a batch of B tokens touches 1 - (1 - active/total)^B of the weights (all of them for dense models).
    const activeFraction = Math.min(1, x.activeParams / x.params);
    const touched = 1 - Math.pow(1 - activeFraction, localBatch);
    const readGB = ((modelTotalGiB * touched + kvTotalGiB) * GiB) / GB;
    const memoryMs = (readGB / (x.bandwidth * speedup * x.bwEff)) * 1000;
    const overheadMs = (x.layers * x.layerOverheadUs) / 1000;
    const computeMs = ((localBatch * 2 * active) / (x.tflops * 1e12 * speedup * H.decodeComputeEff)) * 1000;
    // All-reduce cost relative to the PCIe 4.0 baseline already inside the TP loss above.
    const commRelative = (bw) => Math.pow(12.5 / bw, H.decodeCommBwExponent);
    const commScale = H.decodeCommMsRef * Math.pow(x.tp / 8, H.decodeCommTpExponent) * (x.layers / 32) * Math.pow(x.hidden / 4096, H.decodeCommHiddenExponent);
    const decodeCommDeltaMs = x.tp > 1
      ? commScale * (commRelative(commBandwidth(x, H.interNodeDecodeShare)) - commRelative(H.referenceInterconnectGBs))
      : 0;
    const stepMs = Math.max(memoryMs * 0.5, memoryMs + overheadMs + computeMs + decodeCommDeltaMs);

    const aggregateTPS = ((localBatch * 1000) / stepMs) * replicas;
    // Every active slot and every waiting user shares the aggregate rate.
    const perStream = aggregateTPS / Math.max(1, x.users, x.batch);

    // TTFT = prefill of one prompt (linear + causal attention) + one decode step; independent of users.
    const prompt = Math.min(x.prompt, x.sequence);
    const peak = x.tflops * 1e12 * prefillSpeedup * (x.precision === 'fp8' ? H.fp8PrefillSpeedup : 1);
    const linearMs = ((2 * active * prompt) / (peak * x.computeEff)) * 1000;
    const attentionMs = ((2 * x.layers * x.hidden * prompt * prompt) / (peak * H.prefillAttentionEff)) * 1000;

    const ringBytes = x.tp > 1 ? (2 * x.layers * prompt * x.hidden * 2 * 2 * (x.tp - 1)) / x.tp : 0;
    const prefillCommMs = (bw) => ((H.prefillCommKappa * ringBytes) / (bw * 1e9)) * 1000;
    const queuedPrompts = Math.pow(Math.max(1, x.batch / x.users), H.queuedPrefillExponent);
    const tpSyncMs = x.tp > 2 ? x.layers * H.tpPrefillSyncMsPerLayer * (1 - 2 / x.tp) : 0;
    const prefillCommDeltaMs = tpSyncMs + prefillCommMs(commBandwidth(x, H.interNodePrefillShare)) - prefillCommMs(H.referenceInterconnectGBs);

    return {
      aggregate_tps: aggregateTPS,
      per_stream_tps: perStream,
      ttft_ms: Math.max(0, (linearMs + attentionMs + prefillCommDeltaMs) * queuedPrompts + stepMs),
      decode_ms_per_token: 1000 / Math.max(0.001, perStream),
      step_ms: stepMs,
      bound: memoryMs >= computeMs ? 'memory' : 'compute',
    };
  }

  function trainingMemory(x, ctx) {
    const { localBatch, replicas, stages, weights, overhead, baseBits, totalParams } = ctx;
    const H = HEURISTICS;
    const full = x.ftMethod === 'full';
    // LoRA fraction is a placeholder; an exact module-level inventory would replace it.
    const trainableParams = full ? totalParams : Math.min(totalParams, totalParams * H.loraTrainableFractionAtRank16 * (x.loraRank / 16));
    const trainableGiB = trainableParams / GiB;

    const gradBytes = full && baseBits >= 16 ? bytes(baseBits) : H.lowPrecisionGradientBytes;
    const gradients = trainableGiB * gradBytes;
    const optimizer = trainableGiB * H.optimizerBytesPerParam[x.optimizer];

    // ZeRO partitions across data-parallel replicas, not across TP/PP ranks.
    const zero = Number(x.zero);
    const optShard = zero >= 1 ? replicas : 1;
    const gradShard = zero >= 2 ? replicas : 1;
    const weightShard = zero === 3 ? replicas : 1;

    let activations = (((localBatch * x.sequence * x.layers * x.hidden * 2) / GiB) * (x.flash ? H.activationFactor.flash : H.activationFactor.standard)) / x.pp / x.tp;
    if (x.checkpoint) activations *= H.checkpointActivationScale;

    const logits = (localBatch * x.sequence * H.trainingLogitsGiBPerToken) / x.tp;

    return {
      'Model weights': weights / weightShard,
      Gradients: gradients / stages / gradShard,
      'Optimizer states': optimizer / stages / optShard,
      'Saved activations': activations,
      'Logits / loss buffers': logits,
      'Runtime + communication': overhead + H.trainingFrameworkGiB + (x.ftMethod === 'qlora' ? H.qloraExtraRuntimeGiB : 0),
    };
  }

  function trainingPerformance(x, ctx) {
    const H = HEURISTICS;
    // 6 FLOPs/token/active parameter; LoRA still back-propagates through the frozen base.
    const flopsPerToken = H.trainingFlopsPerParamToken * x.activeParams * 1e9;
    // Small per-device batches under-fill the GPU: efficiency saturates as B / (B + k).
    const fill = ctx.localBatch / (ctx.localBatch + H.trainingBatchSaturation);
    const tokensPerSecond = ((x.tflops * x.gpus * H.trainingMfu * 1e12) / flopsPerToken) * fill;
    return { training_tps: tokensPerSecond, steps_per_second: tokensPerSecond / (x.batch * x.sequence), tokens_per_second: tokensPerSecond };
  }

  function classify(pct) {
    if (pct > 100) return 'insufficient';
    if (pct > 90) return 'very_high';
    if (pct > 75) return 'high';
    if (pct > 50) return 'moderate';
    return 'okay';
  }

  function calculate(x, m = 'inference') {
    validate(x, m);
    const H = HEURISTICS;
    const warnings = [];
    if (x.activeParams > x.params) warnings.push('Active parameter count exceeds total parameters; correct the MoE configuration.');

    const stages = x.tp * x.pp;
    const replicas = x.gpus / stages;
    const localBatch = Math.ceil(x.batch / replicas);
    const totalParams = x.params * 1e9;

    const qlora = m === 'training' && x.ftMethod === 'qlora';
    const format = FORMATS[x.precision];
    const baseBits = qlora ? H.qloraBits : format.bits;
    const metadata = qlora ? H.qloraMetadata : format.overhead;
    const modelTotalGiB = ((totalParams * bytes(baseBits)) / GiB) * (1 + metadata + x.metadata / 100);
    const weights = modelTotalGiB / stages;
    const overhead = H.runtimeBaseGiB[x.runtime] + x.buffers + (stages > 1 ? H.multiGpuOverheadGiB : 0);
    const ctx = { replicas, localBatch, stages, weights, overhead, baseBits, totalParams, modelTotalGiB };

    let components;
    let performance;
    if (m === 'inference') {
      const mem = inferenceMemory(x, ctx);
      components = mem.components;
      performance = inferencePerformance(x, ctx, mem.kvTotalGiB);
      if (x.users > x.batch) warnings.push('Concurrent user demand exceeds active batch; queuing is not reflected in TTFT.');
    } else {
      components = trainingMemory(x, ctx);
      performance = trainingPerformance(x, ctx);
      if (x.zero === '3' && replicas === 1) warnings.push('ZeRO-3 offers no sharding savings without data-parallel replicas.');
      warnings.push('LoRA trainable parameter fraction and activation multipliers are heuristic: use an exact module-level inventory for procurement.');
    }

    const required = sum(components);
    const usable = Math.max(0, x.vram - x.reserve);
    const pct = usable > 0 ? (required / usable) * 100 : Infinity;
    if (pct > 100) warnings.push('Estimated required VRAM exceeds usable GPU memory.');
    if (x.attention === 'mha' && x.kvHeads < 8) warnings.push('Check MHA head count: it generally equals attention query-head count.');

    return {
      mode: m,
      inputs: x,
      components_gib: components,
      required_gib_per_gpu: required,
      usable_gib_per_gpu: usable,
      pct,
      status: classify(pct),
      replicas,
      localBatch,
      performance,
      warnings,
      methodology: {
        weights: 'P × bits/8 × (1 + metadata) / 2^30; distributed by TP×PP; ZeRO-3 shards states across DP replicas',
        kv: 'MHA/GQA: 2 × layers × KV heads × head dim × batch × seq × KV bytes. MLA: layers × (latent + rotary) × batch × seq × KV bytes',
        runtime: 'Framework fixed estimate + configured extra buffers + inter-device allowance',
        performance: 'Bandwidth and FLOPs roofline; efficiency factors adjustable; indicative only',
      },
    };
  }

  return { calculate, validate, classify, PRESETS, GPU_PRESETS, FORMATS, HEURISTICS, GiB };
});
