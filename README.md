# Open LLM Sizing Calculator

> An independent, browser-based **VRAM and performance calculator** for LLM inference and fine-tuning, built from public equations and **validated against 43 live readings** of the ApX calculator. No server, no build, **no dependencies**.

<p align="center">
  <a href="https://github.com/apecia-org/vram-calculator/stargazers"><img src="https://img.shields.io/github/stars/apecia-org/vram-calculator?style=for-the-badge&logo=github&color=ffca28" alt="GitHub stars"></a>
  <a href="https://github.com/apecia-org/vram-calculator/network/members"><img src="https://img.shields.io/github/forks/apecia-org/vram-calculator?style=for-the-badge&logo=github" alt="Forks"></a>
  <a href="https://github.com/apecia-org/vram-calculator/issues"><img src="https://img.shields.io/github/issues/apecia-org/vram-calculator?style=for-the-badge" alt="Issues"></a>
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=for-the-badge" alt="PRs welcome">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=for-the-badge" alt="MIT license"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/node-%E2%89%A518-339933?style=for-the-badge&logo=nodedotjs&logoColor=white" alt="Node 18+">
  <img src="https://img.shields.io/badge/dependencies-0-success?style=for-the-badge" alt="Zero dependencies">
  <img src="https://img.shields.io/badge/tests-37%20passing-success?style=for-the-badge" alt="37 tests passing">
  <img src="https://img.shields.io/badge/median%20decode%20error-2.3%25-blue?style=for-the-badge" alt="Median decode error 2.3%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/JavaScript-ES2020-f7df1e?logo=javascript&logoColor=black" alt="JavaScript">
  <img src="https://img.shields.io/badge/HTML5-single%20page-e34f26?logo=html5&logoColor=white" alt="HTML5">
  <img src="https://img.shields.io/badge/runs-offline-informational" alt="Runs offline">
</p>

<p align="center"><b>Size it. Check it. Ship it.</b></p>

> **Independent project.** Not affiliated with, endorsed by, or derived from the source code of ApX Machine Learning. It uses only publicly published equations, plus numbers read by hand from their public web calculator for validation.

## Why this exists?

Capacity planning for LLMs usually forces a choice between a spreadsheet you can audit and a web tool you can't.

- **Auditable math** — every constant lives in one `HEURISTICS` object, labelled as a fitted heuristic or a published formula.
- **Falsifiable** — each claim is backed by a reading in [`data/apx-readings.json`](./data/apx-readings.json) and a test. Run `npm run validate` to reproduce every number below.
- **Honest about error** — median and 90th-percentile error are published per metric, not one cherry-picked case.
- **Embeddable** — the logic is a pure function with no DOM access: use it in Node, a notebook backend, or a CI check.
- **Offline** — open `index.html`. Nothing leaves your machine.

## Features

- **Inference sizing**: weights, KV cache (MHA, GQA, MLA), workspace and framework overhead, per GPU.
- **Decode speed and TTFT**: bandwidth/compute step model with batching, concurrent users, long-context attention and quadratic prefill attention.
- **Parallelism**: tensor, pipeline and data parallel; NCCL-style overhead; intra- and inter-node interconnect (PCIe, Ethernet).
- **Mixture of experts**: touched-expert model for batch effects (Mixtral, DeepSeek-V3 style).
- **Fine-tuning**: full, LoRA and QLoRA memory with AdamW/8-bit/SGD, ZeRO-1/2/3 sharding, gradient checkpointing and flash attention; calibrated training throughput.
- **Quantization**: FP32, FP16, FP8, INT8, INT4, and GGUF `q4_k_m` / `q5_k_m` / `q8_0`.
- **Exportable**: one-click JSON export and shareable configuration URLs.

## Accuracy

Error is `|ours / ApX − 1|`. ApX readings are not ground truth: several are labelled *Calibrated Benchmark* by ApX and the rest are extrapolated from those anchors. Treat these tables as "close to ApX", not "close to your hardware".

| Metric | Readings | Median abs. error | 90th percentile |
|---|---:|---:|---:|
| Decode tok/s (per user) | 38 | 2.3% | 11.1% |
| TTFT | 38 | 4.8% | 12.7% |
| VRAM per GPU | 31 | 0.7% | 3.7% |
| Training tok/s | 4 | 1.0% | 1.1% |

<sub>Reproduce with `npm run validate`. Raw readings: [`data/apx-readings.json`](./data/apx-readings.json).</sub>

### Per-case comparison

Each cell reads `ApX → ours (difference)`.

<details>
<summary><b>Single-GPU decode, 8B</b></summary>

| Case | Decode tok/s (ApX → ours) | TTFT ms (ApX → ours) | VRAM GiB/GPU (ApX → ours) |
|---|---|---|---|
| 4090 FP16 | 38 → 38 (-0%) | 168 → 161 (-4%) | 15.9 → 16.0 (+1%) |
| 4090 INT4 | 111 → 107 (-4%) | 150 → 144 (-4%) | 4.9 → 4.9 (+1%) |
| 4090 INT4 batch 8, 8 users | 67 → 72 (+8%) | 156 → 148 (-5%) | 5.8 → 5.9 (+2%) |
| H100 PCIe FP16 | 99 → 101 (+2%) | 42 → 39 (-7%) | 17.6 → 17.7 (+1%) |
| H100 PCIe FP8 | 179 → 179 (+0%) | 25 → 22 (-12%) | 10.3 → 10.2 (-1%) |
| H100 PCIe INT8 | 179 → 178 (-1%) | 38 → 35 (-8%) | 10.3 → 10.3 (-0%) |
| H100 PCIe INT4 | 283 → 285 (+1%) | 36 → 33 (-9%) | 6.7 → 6.6 (-1%) |
| H100 SXM FP16 | 156 → 157 (+1%) | 33 → 29 (-13%) | 17.6 → 17.7 (+1%) |
| H100 SXM INT8 | 268 → 262 (-2%) | 30 → 26 (-13%) | 10.3 → 10.3 (-0%) |
| H100 SXM INT4 | 398 → 393 (-1%) | 29 → 25 (-14%) | 6.7 → 6.6 (-1%) |
| A100 40GB FP16 | 79 → 80 (+1%) | 81 → 84 (+3%) | 17.6 → 17.7 (+1%) |
| A100 40GB INT8 | 145 → 142 (-2%) | 75 → 78 (+4%) | 10.3 → 10.3 (-0%) |
| A100 40GB INT4 | 235 → 232 (-1%) | 72 → 75 (+5%) | 6.7 → 6.6 (-1%) |

</details>

<details>
<summary><b>Batching, 8B on A100 40GB</b></summary>

| Case | Decode tok/s (ApX → ours) | TTFT ms (ApX → ours) | VRAM GiB/GPU (ApX → ours) |
|---|---|---|---|
| FP16 batch 8 | 69 → 67 (-2%) | 82 → 86 (+5%) | 18.4 → 18.7 (+2%) |
| INT8 batch 8 | 114 → 107 (-6%) | 77 → 81 (+5%) | 11.2 → 11.3 (+1%) |
| INT4 batch 8 | 153 → 151 (-1%) | 75 → 78 (+4%) | 7.6 → 7.6 (+1%) |

</details>

<details>
<summary><b>Long context, 8B FP16 on A100 40GB</b></summary>

| Case | Decode tok/s (ApX → ours) | TTFT ms (ApX → ours) | VRAM GiB/GPU (ApX → ours) |
|---|---|---|---|
| 8K tokens | 75 → 76 (+1%) | 565 → 632 (+12%) | 18.5 → 18.6 (+0%) |
| 32K tokens | 64 → 65 (+1%) | 2970 → 3167 (+7%) | 21.5 → 21.6 (+0%) |
| 64K tokens | 53 → 54 (+3%) | 7990 → 8125 (+2%) | 25.5 → 25.6 (+0%) |

</details>

<details>
<summary><b>Multi-GPU, 70B INT4 on A100 40GB (PCIe 4.0)</b></summary>

| Case | Decode tok/s (ApX → ours) | TTFT ms (ApX → ours) | VRAM GiB/GPU (ApX → ours) |
|---|---|---|---|
| 1 GPU | 31 → 33 (+6%) | 671 → 652 (-3%) | 36.6 → 37.1 (+1%) |
| TP2 | 62 → 60 (-3%) | 339 → 328 (-3%) | 21.0 → 20.7 (-2%) |
| TP4 | 96 → 98 (+2%) | 208 → 200 (-4%) | 12.5 → 12.0 (-4%) |
| TP8 | 138 → 148 (+8%) | 140 → 135 (-3%) | 8.2 → 7.7 (-6%) |
| PP4 | 34 → 36 (+5%) | 575 → 599 (+4%) | 12.9 → 12.0 (-7%) |
| TP2 × PP4 | 60 → 65 (+8%) | 339 → 301 (-11%) | 8.5 → 7.7 (-9%) |

</details>

<details>
<summary><b>Interconnect, TP8 on 8× A100 40GB</b></summary>

| Case | Decode tok/s (ApX → ours) | TTFT ms (ApX → ours) | VRAM GiB/GPU (ApX → ours) |
|---|---|---|---|
| 8B, PCIe 4.0 (64 GB/s) | 579 → 643 (+11%) | 27 → 31 (+14%) | — |
| 8B, PCIe 3.0 (32 GB/s) | 534 → 598 (+12%) | 36 → 36 (+1%) | — |
| 8B, 100 GbE (12.5 GB/s) | 383 → 415 (+8%) | 50 → 53 (+7%) | — |
| 8B, 10 GbE (1.25 GB/s) | 17 → 17 (+1%) | 315 → 353 (+12%) | — |
| 70B, 100 GbE (12.5 GB/s) | 68 → 72 (+6%) | 240 → 252 (+5%) | — |
| 70B, 10 GbE (1.25 GB/s) | 2 → 2 (+3%) | 1700 → 1939 (+14%) | — |
| 70B TP4, 100 GbE | 74 → 74 (-0%) | 291 → 296 (+2%) | — |
| 70B TP2, 100 GbE | 56 → 55 (-2%) | 401 → 391 (-2%) | — |
| 70B, 2 nodes × 4 GPUs, PCIe 4.0 + 100 GbE | 101 → 107 (+6%) | 229 → 236 (+3%) | — |

</details>

<details>
<summary><b>Mixture of experts</b></summary>

| Case | Decode tok/s (ApX → ours) | TTFT ms (ApX → ours) | VRAM GiB/GPU (ApX → ours) |
|---|---|---|---|
| Mixtral 8x7B INT4, 1× A100 40GB, batch 1 | 259 → 256 (-1%) | 64 → 66 (+3%) | 25.1 → 25.4 (+1%) |
| Mixtral 8x7B INT4, 1× A100 40GB, batch 8 | 70 → 64 (-9%) | 449 → 420 (-7%) | — |
| DeepSeek-V3 INT4, 8× H100, TP8, batch 1 | 397 → 331 (-17%) | 54 → 55 (+1%) | — |
| DeepSeek-V3 INT4, 8× H100, TP8, batch 8 | 97 → 119 (+22%) | 325 → 345 (+6%) | — |

</details>

<details>
<summary><b>Fine-tuning, Llama 3.1 8B on H100 80GB</b></summary>

| Case | VRAM GiB (ApX → ours) | Training tok/s (ApX → ours) |
|---|---|---|
| Full FP16, batch 1 | 129 → 129 (+0%) | — |
| LoRA r16, batch 1 | 24.2 → 24.5 (+1%) | 5729 → 5675 (-1%) |
| LoRA r16, batch 8 | 66.4 → 66.5 (+0%) | 8188 → 8099 (-1%) |
| QLoRA, batch 1 | 13.5 → 14.0 (+4%) | 5729 → 5675 (-1%) |
| QLoRA, batch 8 | 55.6 → 56.0 (+1%) | 8188 → 8099 (-1%) |

</details>

### Coverage and known gaps

| Area | Covered | Not covered yet |
|---|---|---|
| Models | Llama 3.1 8B / 70B, Mixtral 8x7B, DeepSeek-V3 | Dense models above 70B, VLMs, speculative decoding |
| GPUs | RTX 4090, A100 40 GB, H100 PCIe/SXM | AMD, Apple silicon, H200 profile is **unmeasured** |
| Fine-tuning | Llama 8B on one H100 (Full, LoRA, QLoRA) | Other GPUs, multi-GPU ZeRO |
| Interconnect | PCIe 3.0/4.0, 10/100 GbE, 2-node TP8 | NVLink / NVSwitch |
| Worst cases | DeepSeek-V3 decode (−17% / +22%), 8B TP8 decode (+11%) | — |

## Quick start

### In the browser

```sh
git clone https://github.com/apecia-org/vram-calculator.git
cd vram-calculator
open index.html        # or: python3 -m http.server and visit http://localhost:8000
```

### From Node (18+)

```js
const { calculate, PRESETS, GPU_PRESETS } = require('./calculator.js');

const result = calculate({
  ...PRESETS.llama70,        // 70B, GQA
  ...GPU_PRESETS.a100,       // 80 GB, 2039 GB/s, 312 TFLOPS + fitted efficiencies
  precision: 'int4', kvPrecision: 'fp16',
  sequence: 8192, prompt: 1024, batch: 1, users: 1,
  gpus: 4, tp: 4, pp: 1, gpusPerNode: 8, interconnectGBs: 64, interNodeGBs: 12.5,
  reserve: 1.5, metadata: 0, runtime: 'vllm', computeEff: 0.75,
  loraRank: 16, ftMethod: 'full', optimizer: 'adamw', zero: '0',
  checkpoint: false, flash: true, latent: 512, rotary: 64,
});

console.log(result.status, result.required_gib_per_gpu, result.performance);
```

A complete runnable version is in [`examples/node-api.js`](./examples/node-api.js).

```sh
npm test            # unit + regression tests (Node's built-in runner)
npm run validate    # re-create the accuracy tables from data/apx-readings.json
```

## Project layout

```
calculator.js            → pure sizing logic (no DOM): window.LLMSizing / require()
app.js                   → DOM wiring for index.html
index.html               → single-page UI, no build step
data/apx-readings.json   → every ApX reading used for fitting and validation
scripts/validate.js      → compares calculator.js to the readings, prints the tables
examples/node-api.js     → minimal Node usage
test/calculator.test.js  → formula, invariant and regression tests
task.md                  → original working notes and formula list
```

## How it works

| Quantity | Model |
|---|---|
| Weights | `params × bits/8 × (1 + metadata) / 2³⁰`, divided by TP × PP |
| KV cache | MHA/GQA: `2 × layers × kv_heads × head_dim × batch × tokens × bytes`; MLA: `layers × (latent + rotary) × …` |
| Decode step | `(weights·touched + batch·KV)/(BW·eff) + layers·overhead + batch·2·active/(FLOPs·0.215)` |
| Per-user speed | `aggregate / max(users, batch)`; every sequence emits one token per step |
| TTFT | `2·active·N/(peak·MFU) + 2·layers·hidden·N²/peak` + one decode step + TP sync |
| Tensor parallel | `tp / (1 + loss·doublings)` plus an interconnect term relative to PCIe 4.0 |
| Pipeline parallel | `1 + 0.03·(pp−1)` single-stream gain: stages run sequentially |
| MoE | batch of B tokens reads `1 − (1 − active/total)^B` of the weights |
| Fine-tuning | weights + gradients + 12 B/param AdamW states (ZeRO-sharded across data-parallel replicas) + activations + logits |
| Training speed | `FLOPs·0.42 / (6·params) × B/(B+0.52)` |

The structural terms (weights, KV, FLOPs) come from published equations. The efficiency constants come from fitting ApX readings and are listed in `HEURISTICS` and `GPU_PRESETS`.

## Configuration

GPU presets carry fitted efficiency profiles. Override any field per call or in the UI.

| Preset | VRAM | Bandwidth | TFLOPS | `bwEff` | Per-layer overhead | Buffers |
|---|---:|---:|---:|---:|---:|---:|
| `4090` | 24 GB | 1008 GB/s | 165 | 0.69 | 84 µs | 0.3 GiB |
| `a100` | 80 GB | 2039 GB/s | 312 | 0.93 | 34 µs | 2.0 GiB |
| `h100` | 80 GB | 3350 GB/s | 989 | 0.93 | 34 µs | 2.0 GiB |
| `h200` | 141 GB | 4800 GB/s | 989 | 0.93 | 34 µs | 2.0 GiB (profile **unmeasured**) |

## Tech stack

| Layer | Tools |
|---|---|
| Logic | Plain JavaScript (UMD module), no dependencies |
| UI | Single HTML page, vanilla DOM |
| Tests | `node:test`, `node:assert` |
| Data | JSON readings with provenance |

## Roadmap

| Phase | What it ships | Status |
|---|---|---|
| 1 | Weights, KV cache, basic roofline | ✅ Done |
| 2 | Calibrated decode, TTFT, batching, long context | ✅ Done |
| 3 | Tensor/pipeline parallel, interconnects, multi-node | ✅ Done |
| 4 | MoE, fine-tuning (Full/LoRA/QLoRA on H100) | ✅ Done |
| 5 | Fine-tuning on other GPUs, multi-GPU ZeRO | ⏳ Planned |
| 6 | NVLink, Apple unified memory, speculative decoding | ⏳ Planned |
| 7 | Parse Hugging Face `config.json` for exact architectures and LoRA module counts | ⏳ Planned |

## Contributing

1. Fork and clone the repository.
2. Run `npm test` and `npm run validate` to see the baseline.
3. Add readings to `data/apx-readings.json` (or measured hardware numbers) and the code change that explains them.
4. Open a pull request. Include the before/after output of `npm run validate`.

The most useful contribution is **new measured data**: other GPUs, models and parallel layouts.

## Limitations

- Estimates, not guarantees of out-of-memory avoidance or throughput.
- Efficiency constants are fitted to ApX, not measured on hardware.
- Activations and framework overhead are heuristic proxies; use a real profile before procurement.
- No CPU/NVMe offload simulation.

## Show your support

If this saved you a spreadsheet, the cheapest way to say thanks is a star.

<p align="center">
  <a href="https://github.com/apecia-org/vram-calculator"><img src="https://img.shields.io/github/stars/apecia-org/vram-calculator?style=social" alt="Star on GitHub"></a>
</p>

## License

[MIT](./LICENSE) — free forever. Use it, fork it, ship it; keep the copyright notice.
