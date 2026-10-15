# Open LLM Sizing Calculator

An independent, browser-based LLM VRAM and performance sizing calculator based on public equations published by ApX Machine Learning. This is **not** ApX source code and does not reproduce its private benchmarks, calibration coefficients, underlying databases or paid API.

## Run

Open `index.html` in a modern browser. No build, packages, server or internet needed.

## Layout

- `calculator.js` — pure sizing logic (no DOM). Loaded by the page as `window.LLMSizing` and importable in Node via `require('./calculator.js')`. All heuristic constants live in its `HEURISTICS` object.
- `app.js` — DOM wiring: reads the form, calls `calculate()`, renders results.
- `index.html` — markup and styles.
- `test/calculator.test.js` — unit tests for the formulas (`npm test`, Node 18+).

## Public formulas implemented

1. Model weights in GiB: `total_parameters * (weight_bits/8) * (1 + quantization_metadata_fraction) / 2**30`
2. MHA / GQA KV in bytes: `2 * layers * KV_heads * head_dim * local_batch * context_tokens * bytes_per_KV_element`.
3. MLA KV in bytes: `layers * (latent_dim + rotary_dim) * local_batch * context_tokens * bytes_per_KV_element`.
4. Parallel weights/KV on each active device: approximate division by `TP * PP` (partition assumptions may differ by runtime).
5. Full fine-tuning: resident weights, trainable gradients, optimizer states (12 B/param for AdamW), activations (~38 B per token per layer per hidden unit with flash attention), logits/loss buffers and framework buffers. Throughput: `TFLOPS*0.42/(6*params) * B/(B+0.52)` (fit to ApX H100 HF-trainer readings).
6. LoRA / QLoRA: frozen backbone and adapter gradients/optimizer; adapter count is a *placeholder percentage*, not a module-specific calculation.
7. Decode step (per replica): `(weights + batch*KV)/(BW*speedup*bwEff) + layers*layerOverhead + batch*2*active/(TFLOPS*speedup*0.215)`, where `speedup = TP*(1+0.03*(PP-1))`; each sequence emits one token per step and per-user speed = aggregate / max(users, batch). `bwEff` and `layerOverhead` come from the GPU preset (fitted to ApX readings: HBM ≈ 0.93 / 34 µs, GDDR ≈ 0.69 / 84 µs). Prefill: `2*active*N/(peak*computeEff) + 2*layers*hidden*N²/peak`, computeEff ≈ 0.75; TTFT = prefill + one decode step (batch prompts shared by one user prefill together).
8. Interconnect: TP all-reduce cost relative to PCIe 4.0 (64 GB/s) scales as `bandwidth^-1.8` for decode and `1/bandwidth` for prefill; TP groups spanning nodes blend intra- and inter-node bandwidth. MoE: a batch of B tokens reads `1-(1-active/total)^B` of the weights.
9. Hardware-fit status thresholds: ≤50% okay; ≤75% moderate; ≤90% high; ≤100% very high; >100% insufficient.

## Assumptions and important limitations

- Published mathematical model is independently implemented; runtime memory, model formats, GPU specs, efficiency values and quantization metadata are adjustable, **not exact ApX internals**.
- GPU presets and architectural defaults are illustrative; source-of-truth should be GPU vendor specs and the specific model `config.json`.
- Decode estimates are heuristic. Batch use, memory bandwidth saturation, TP/PP communication and actual cache-block allocation vary substantially.
- The tool uses GiB (2^30 bytes) for all memory capacities; verify whether GPU quoted GB is decimal or binary.
- ZeRO stages are simplified and only shard across data-parallel replicas. The calculator does not implement FSDP, activation offloading, CPU/NVMe offloading, packing, Mamba or attention hybridization.
- Training activation and adapter estimates require validation on a measured training run.
- A physical benchmark with vLLM, TensorRT-LLM or target training stack remains necessary before procurement.

## Primary public sources

- https://apxml.com/tools/vram-calculator
- https://apxml.com/docs/calculator

## Suggested next development steps

- Parse Hugging Face `config.json` to derive exact architecture and LoRA trainable parameters.
- Add calibrated runtime profiles with benchmark results for chosen GPU + model + runner combinations.
- Integrate a constraint-based GPU setup planner and simultaneous request-arrival/queuing simulation.
- Add a Python port of `calculate()` if a non-JS API is needed (the Node API and unit tests exist).
