# Compression and quantization in PowerInfer

This document covers three questions raised during the PowerInfer modernization work:

1. Whether TurboQuant-style compression fits the current architecture and sparse-routing assumptions.
2. Which tensors and compute paths are the highest-value targets for future compression experiments.
3. Which compression directions are production-ready now versus research-only.

---

## Architecture context

PowerInfer divides FFN computation across two regimes at load time:

- **Hot path (GPU)**: A per-layer subset of neurons whose activation statistics exceed the sparse threshold are copied into striped matrices (`ffn_gate_gpu`, `ffn_up_gpu`, `ffn_down_gpu`) resident on the GPU. The stripe layout is derived from `gpu_idx` and `gpu_bucket`, which are built from pre-computed activation files.
- **Cold path (CPU)**: The remaining neurons stay in main memory and are computed on the CPU for the tokens that the predictor says need them.
- **Predictor (GPU)**: A lightweight two-layer ReLU MLP (`mlp_pre_w1`, `mlp_pre_w2`) runs on the GPU first to decide which neurons fall in the hot versus cold bucket for each forward pass.

This split is the critical backdrop for any compression discussion. A technique that makes sense for a uniform dense model may have very different cost/quality tradeoffs when one part of the same weight matrix lives on GPU and the other lives on CPU.

---

## TurboQuant-style compression fit

"TurboQuant-style" refers broadly to aggressive post-training quantization that combines block-wise scale calibration, outlier-aware rounding, and optional mixed-precision assignment to stay below 4 bits per weight without large accuracy loss.

### Where it fits

**Cold FFN weights on CPU**

The CPU-resident FFN slices are pure memory-bandwidth consumers. For a 7B model with 32 layers the cold portion of `ffn_up` and `ffn_down` alone can be several gigabytes. A 2-bit or 3-bit scheme reduces the memory footprint and bandwidth cost of cold-path reads proportionally. The activation statistics that determine the hot/cold split are computed from model behavior, not from weight values, so compressing the weights does not invalidate the routing decisions. This is the highest-leverage application of aggressive quantization in the entire stack.

**Hot GPU FFN slices**

The GPU striped matrices are materialized at load time by slicing columns from the full quantized weight tensors. Whatever quantization type is stored in the GGUF file propagates directly into the GPU slices. Standard INT4 dequantization on GPU (Q4_0, Q4_K_M) is already fast and well-supported by CUDA kernels. Pushing below 4 bits on GPU is possible but dequantization overhead per token grows if the kernel path is not optimized for sub-4-bit formats, and the hot slice is small enough that memory pressure is already relieved by the sparsity itself. Mixed-precision that applies lower bits to cold weights and keeps hot slices at Q4_K or Q5_K is a reasonable middle ground.

**Attention tensors**

Attention projections (`attn_q`, `attn_k`, `attn_v`, `attn_output`) are always fully computed and follow no sparse routing. They behave identically to dense-model tensors and respond well to k-quant formats (Q4_K_M, Q5_K_M). TurboQuant-style calibration could further reduce attention memory, but these tensors are not the bottleneck that sparse routing was designed to address. Standard k-quants are already adequate here.

**Output logits tensor**

The `output.weight` tensor maps the final hidden state to vocabulary logits. Quantization noise here has an outsized effect on generation quality and perplexity. The existing `--leave-output-tensor` flag exists for exactly this reason and should remain the default unless careful per-task evaluation shows a particular format is safe.

### Where it does not fit cleanly

**Predictor weights**

`mlp_pre_w1` and `mlp_pre_w2` are the sparse-routing engine. They are small (a two-layer MLP with a hidden dimension much smaller than the main FFN), so the memory benefit of aggressively quantizing them is minimal. More importantly, rounding errors in these weights change which neurons are classified as hot or cold each token, which cascades into every downstream FFN computation. Predictor compression should not be treated as equivalent to FFN weight compression. Any quantization applied to predictor weights requires direct routing-accuracy measurement before deployment.

**GPU index tensors**

`gpu_idx` and `gpu_bucket` are integer index arrays, not learned weights. They should not be quantized.

### Summary

TurboQuant-style compression is architecturally compatible with PowerInfer's sparse-routing design provided it is applied to FFN and attention weights rather than to predictor weights or index structures. The split between hot and cold regimes makes mixed-precision a natural fit: lower bits for cold CPU-resident weights, standard INT4 for hot GPU-resident slices, and unchanged predictor weights.

---

## Highest-value tensors and paths for future experiments

Ranked by expected impact (memory reduction × frequency of compute):

| Rank | Tensor group | Regime | Why high value |
| --- | --- | --- | --- |
| 1 | Cold FFN slice (`ffn_up`, `ffn_down`, `ffn_gate` columns not in GPU bucket) | CPU | Largest total bytes, bandwidth-bound, compression does not affect routing decisions |
| 2 | Attention projections (`attn_q`, `attn_k`, `attn_v`, `attn_output`) | GPU / CPU split | Always computed, large, standard k-quants apply cleanly |
| 3 | Hot GPU FFN slice (`ffn_gate_gpu`, `ffn_up_gpu`, `ffn_down_gpu`) | GPU | Already small due to sparsity; INT4 is the practical floor unless sub-4-bit GPU kernels are added |
| 4 | Token embedding (`token_embd`) | CPU | Large lookup table, accessed once per token per forward pass, quantization-friendly |
| 5 | Predictor (`mlp_pre_w1`, `mlp_pre_w2`) | GPU | Small in size; compression is risky because routing accuracy depends on it |

### Paths worth evaluating

- **Mixed-precision GGUF**: Produce a GGUF where cold FFN columns use Q2_K or Q3_K_S and hot columns use Q4_K_M or Q5_K_M. The current `quantize` binary applies a single format to each tensor. A mixed-precision conversion script would need to slice the full-precision weights before quantizing each part.
- **Per-layer compression rates**: Layers near the beginning and end of the network often tolerate lower precision than middle layers. A per-layer sensitivity scan (perplexity delta per layer as a function of quant level) would identify which layers can be compressed most aggressively.
- **Predictor distillation as a compression proxy**: Instead of quantizing predictor weights, train a smaller predictor and measure routing recall. This avoids rounding noise in the routing path entirely.
- **KV cache compression**: Not covered by the GGUF quantization path but relevant to memory pressure during long-context inference. Falls under a separate engineering track.

---

## Production-ready versus research-only

### Production-ready now

These use existing tooling (`quantize` binary, existing GGUF conversion) with no architectural changes:

- **Standard k-quants for full-model quantization.** Q4_K_M is the recommended default for desktop deployment. Q5_K_M is a safe step up when VRAM allows. Q3_K_M is usable for memory-constrained hardware and shows acceptable perplexity at this model class.

  ```bash
  ./build/bin/quantize /PATH/TO/MODEL.powerinfer.gguf /PATH/TO/MODEL.q4km.powerinfer.gguf Q4_K_M
  ```

- **Leaving the output tensor unquantized** with `--leave-output-tensor` when requantizing from an already-quantized source to protect logit quality.

- **Choosing quant level by VRAM budget.** The `--vram-budget` flag already controls GPU offload. A lower quant level (Q3_K, Q4_0) reduces memory footprint of both the hot GPU slice and the resident model portions proportionally.

### Research exploration

These require experimentation before production use and are not yet wired into the standard PowerInfer conversion or inference path:

- **Mixed hot/cold precision**: Requires a custom conversion script that identifies the hot bucket per layer before quantizing and applies different quant levels to each slice. Needs end-to-end perplexity and throughput validation.
- **Sub-3-bit cold FFN compression**: Q2_K is already supported by the GGUF quantization path. Below 2.5 bits per weight (e.g., 1-bit or ternary weight schemes) requires new kernel support and careful evaluation against PowerInfer's ReLU-sparse model families.
- **Predictor compression**: Any change to predictor precision must be benchmarked against routing recall (fraction of truly active neurons that the predictor correctly identifies) rather than just perplexity. A small routing recall drop can cascade into large output quality drops.
- **Structured pruning of cold weights**: Instead of quantizing cold neurons, zeroing out the lowest-magnitude cold columns and compressing the remaining dense sub-matrix. Intersects with sparsity-aware training work (TurboSparse) rather than post-training quantization alone.
- **KV cache quantization**: Compressing the key/value cache at INT8 or lower to reduce memory during long-context inference. No current runtime support in this codebase.
