# PowerInfer: sparse LLM inference for practical local deployment

PowerInfer is an inference engine for running large language models on local hardware by exploiting **activation locality**. Instead of treating every neuron as equally active, PowerInfer separates consistently hot paths from input-dependent cold paths so sparse models can use GPU memory, CPU compute, and VRAM budgets more efficiently.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## What this repository is for

This repository now covers two closely related deployment tracks:

- **PowerInfer classic**: hybrid CPU/GPU inference for ReLU-sparse and PowerInfer GGUF models on desktops and workstations.
- **SmallThinker**: the newer on-device sparse MoE direction for phones, edge devices, and constrained local deployments, documented under [`smallthinker/`](./smallthinker/README.md).

If you need the shortest positioning summary:

- **Activation locality** is the core idea behind PowerInfer.
- **Sparse inference** is how that idea turns into practical speed and memory wins.
- **SmallThinker** is the current flagship model family for on-device deployment in this repo.
- **Practical deployment** means buildable local tooling, reproducible conversion flows, and examples for CLI, serving, evaluation, and adapters.

## Why PowerInfer still matters

PowerInfer was built around a simple observation: during inference, only a subset of the model is consistently important for each token. By keeping the hot paths close to the GPU and handling colder paths more selectively, PowerInfer reduces memory pressure and unnecessary movement while preserving a familiar local-LLM workflow.

That makes the project useful when you want to:

- run sparse models on a single consumer GPU,
- control GPU usage with a predictable `--vram-budget`,
- keep a local-first stack for CLI, batch, and server workflows,
- explore sparse model deployment rather than dense-only inference,
- move toward newer on-device sparse deployments through SmallThinker.

## Project map

| Area | Use it for |
| --- | --- |
| [Root PowerInfer runtime](./README.md) | Classic PowerInfer build, model conversion, quantization, sparse inference, and hybrid CPU/GPU deployment |
| [`smallthinker/`](./smallthinker/README.md) | Current SmallThinker models and the on-device sparse MoE runtime |
| [`examples/`](./examples) | CLI, server, batched generation, perplexity, quantization, and fine-tuning examples |
| [`docs/`](./docs) | Root-project troubleshooting, backend notes, and compression/quantization guidance |
| [`smallthinker/docs/`](./smallthinker/docs) | Install, build, function calling, multimodal, and backend-specific SmallThinker docs |
| [`gguf-py/`](./gguf-py/README.md) | GGUF file tooling and Python packaging |
| [`powerinfer-py/`](./powerinfer-py) | Python helpers for PowerInfer runtime workflows |
| [`grammars/`](./grammars/README.md) | Grammar-constrained generation assets |

## Current highlights

- **SmallThinker**: [SmallThinker 21B-A3B Instruct](https://huggingface.co/PowerInfer/SmallThinker-21BA3B-Instruct) and [SmallThinker 4B-A0.6B Instruct](https://huggingface.co/PowerInfer/SmallThinker-4BA0.6B-Instruct) are the clearest view of the repo's current sparse, on-device direction.
- **PowerInfer-2**: [paper](https://arxiv.org/abs/2406.06282) on smartphone-oriented sparse inference.
- **Turbo Sparse**: [paper](https://arxiv.org/abs/2406.05955) on training highly sparse models for efficient deployment.
- **Original PowerInfer paper**: [arXiv:2312.12456](https://arxiv.org/abs/2312.12456).

## Demo

https://github.com/SJTU-IPADS/PowerInfer/assets/34213478/fe441a42-5fce-448b-a3e5-ea4abb43ba23

PowerInfer versus llama.cpp on a single RTX 4090 running Falcon(ReLU)-40B-FP16 showed up to an 11x token-generation speedup in the original release evaluation.

## Deployment paths

### 1. Desktop sparse inference with PowerInfer
Use the root project when you want classic PowerInfer GGUF flows, hybrid CPU/GPU execution, quantization, and the familiar `main` / `server` / `batched` style tooling.

### 2. Dense compatibility workflows
PowerInfer can still run compatible dense GGUF variants for some workflows, but the main value of this repository is sparse inference and sparse-model deployment.

### 3. On-device sparse MoE with SmallThinker
Use [`smallthinker/`](./smallthinker/README.md) when you want the repo's newer path for phones, edge devices, or strict memory budgets.

## Getting started

- [5-minute quickstart](#5-minute-quickstart)
- [Build the root project](#build)
- [Get model weights](#model-weights)
- [Run inference](#inference)
- [Serve or benchmark a model](#serving-evaluation-and-other-workflows)
- [Server API and agent integration](#server-api-and-agent-integration)
- [Benchmarks and validation](#benchmarks-and-validation)
- [Fine-tune supported models](#fine-tuning-addendum)
- [Explore SmallThinker](./smallthinker/README.md)

## Prerequisites

PowerInfer requires:

- CMake 3.17+
- Python 3.8+ and pip
- a supported compiler toolchain for your platform
- optional GPU backend dependencies for CUDA or ROCm builds

## Get the code

```bash
git clone <your PowerInfer repository URL>
cd PowerInfer
pip install -r requirements.txt
```

## 5-minute quickstart

This path gets you from a fresh clone to a verified running model with the smallest possible model on CPU-only hardware. GPU builds are faster but this path requires nothing beyond a C++ compiler and Python.

**Step 1 – build (CPU-only)**

```bash
cmake -S . -B build
cmake --build build --config Release -j$(nproc)
```

**Step 2 – download the smallest PowerInfer GGUF**

```bash
huggingface-cli download --resume-download \
  --local-dir ReluLLaMA-7B \
  --local-dir-use-symlinks False \
  PowerInfer/ReluLLaMA-7B-PowerInfer-GGUF
```

**Step 3 – run your first inference**

```bash
./build/bin/main \
  -m ReluLLaMA-7B/*.powerinfer.gguf \
  -n 64 \
  -t "$(nproc)" \
  -p "Once upon a time"
```

**Step 4 – verify output quality (perplexity spot check)**

```bash
./build/bin/perplexity \
  -m ReluLLaMA-7B/*.powerinfer.gguf \
  -f grammars/README.md \
  --chunks 1
```

A successful run prints a perplexity value and timing. Any finite number means the model loaded and ran correctly.

**Step 5 – start the server**

```bash
./build/bin/server \
  -m ReluLLaMA-7B/*.powerinfer.gguf \
  -c 2048 \
  --port 8080
```

Then from a second shell:

```bash
curl -s http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Once upon a time","n_predict":32}' \
  | python3 -m json.tool
```

If you see a `content` field in the JSON, the stack is working end-to-end.

## Build

These commands are run from the repository root.

### NVIDIA GPU

```bash
cmake -S . -B build -DLLAMA_CUBLAS=ON
cmake --build build --config Release
```

### AMD GPU

```bash
# Replace '1100' with your GPU architecture from rocminfo
CC=/opt/rocm/llvm/bin/clang CXX=/opt/rocm/llvm/bin/clang++ cmake -S . -B build -DLLAMA_HIPBLAS=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build --config Release
```

### CPU only

```bash
cmake -S . -B build
cmake --build build --config Release
```

## Model weights

PowerInfer models are stored as **PowerInfer GGUF** files, which bundle the model weights with predictor data used for sparse routing and FFN offloading.

### Download PowerInfer GGUF from Hugging Face

| Base model | PowerInfer GGUF |
| --- | --- |
| LLaMA(ReLU)-2-7B | [PowerInfer/ReluLLaMA-7B-PowerInfer-GGUF](https://huggingface.co/PowerInfer/ReluLLaMA-7B-PowerInfer-GGUF) |
| LLaMA(ReLU)-2-13B | [PowerInfer/ReluLLaMA-13B-PowerInfer-GGUF](https://huggingface.co/PowerInfer/ReluLLaMA-13B-PowerInfer-GGUF) |
| Falcon(ReLU)-40B | [PowerInfer/ReluFalcon-40B-PowerInfer-GGUF](https://huggingface.co/PowerInfer/ReluFalcon-40B-PowerInfer-GGUF) |
| LLaMA(ReLU)-2-70B | [PowerInfer/ReluLLaMA-70B-PowerInfer-GGUF](https://huggingface.co/PowerInfer/ReluLLaMA-70B-PowerInfer-GGUF) |
| ProSparse-LLaMA-2-7B | [PowerInfer/ProSparse-LLaMA-2-7B-GGUF](https://huggingface.co/PowerInfer/prosparse-llama-2-7b-gguf) |
| ProSparse-LLaMA-2-13B | [PowerInfer/ProSparse-LLaMA-2-13B-GGUF](https://huggingface.co/PowerInfer/prosparse-llama-2-13b-gguf) |
| Bamboo-base-7B | [PowerInfer/Bamboo-base-v0.1-gguf](https://huggingface.co/PowerInfer/Bamboo-base-v0.1-gguf) |
| Bamboo-DPO-7B | [PowerInfer/Bamboo-DPO-v0.1-gguf](https://huggingface.co/PowerInfer/Bamboo-DPO-v0.1-gguf) |

We recommend `huggingface-cli` for full-repo downloads:

```bash
huggingface-cli download --resume-download --local-dir ReluLLaMA-7B --local-dir-use-symlinks False PowerInfer/ReluLLaMA-7B-PowerInfer-GGUF
```

Expected layout:

```text
.
├── *.powerinfer.gguf
├── *.q4.powerinfer.gguf
├── activation
│   ├── activation_x.pt
│   └── ...
└── *.[q4].powerinfer.gguf.generated.gpuidx
```

### Convert original weights plus predictor weights

For large unquantized models, you can convert original weights plus predictor weights into PowerInfer GGUF.

| Base model | Original model | Predictor |
| --- | --- | --- |
| LLaMA(ReLU)-2-7B | [SparseLLM/ReluLLaMA-7B](https://huggingface.co/SparseLLM/ReluLLaMA-7B) | [PowerInfer/ReluLLaMA-7B-Predictor](https://huggingface.co/PowerInfer/ReluLLaMA-7B-Predictor) |
| LLaMA(ReLU)-2-13B | [SparseLLM/ReluLLaMA-13B](https://huggingface.co/SparseLLM/ReluLLaMA-13B) | [PowerInfer/ReluLLaMA-13B-Predictor](https://huggingface.co/PowerInfer/ReluLLaMA-13B-Predictor) |
| Falcon(ReLU)-40B | [SparseLLM/ReluFalcon-40B](https://huggingface.co/SparseLLM/ReluFalcon-40B) | [PowerInfer/ReluFalcon-40B-Predictor](https://huggingface.co/PowerInfer/ReluFalcon-40B-Predictor) |
| LLaMA(ReLU)-2-70B | [SparseLLM/ReluLLaMA-70B](https://huggingface.co/SparseLLM/ReluLLaMA-70B) | [PowerInfer/ReluLLaMA-70B-Predictor](https://huggingface.co/PowerInfer/ReluLLaMA-70B-Predictor) |
| ProSparse-LLaMA-2-7B | [SparseLLM/ProSparse-LLaMA-2-7B](https://huggingface.co/SparseLLM/prosparse-llama-2-7b) | [PowerInfer/ProSparse-LLaMA-2-7B-Predictor](https://huggingface.co/PowerInfer/prosparse-llama-2-7b-predictor) |
| ProSparse-LLaMA-2-13B | [SparseLLM/ProSparse-LLaMA-2-13B](https://huggingface.co/SparseLLM/prosparse-llama-2-13b) | [PowerInfer/ProSparse-LLaMA-2-13B-Predictor](https://huggingface.co/PowerInfer/prosparse-llama-2-13b-predictor) |
| Bamboo-base-7B | [PowerInfer/Bamboo-base-v0.1](https://huggingface.co/PowerInfer/Bamboo-base-v0_1) | [PowerInfer/Bamboo-base-v0.1-predictor](https://huggingface.co/PowerInfer/Bamboo-base-v0.1-predictor) |
| Bamboo-DPO-7B | [PowerInfer/Bamboo-DPO-v0.1](https://huggingface.co/PowerInfer/Bamboo-DPO-v0_1) | [PowerInfer/Bamboo-DPO-v0.1-predictor](https://huggingface.co/PowerInfer/Bamboo-DPO-v0.1-predictor) |

```bash
python convert.py --outfile /PATH/TO/POWERINFER/GGUF/REPO/MODELNAME.powerinfer.gguf /PATH/TO/ORIGINAL/MODEL /PATH/TO/PREDICTOR
```

If you need a dense GGUF variant for compatibility-oriented workflows:

```bash
python convert-dense.py --outfile /PATH/TO/DENSE/GGUF/REPO/MODELNAME.gguf /PATH/TO/ORIGINAL/MODEL
```

Dense exports can be useful for compatibility and fine-tuning workflows, but they do not provide the sparse inference benefits that define PowerInfer.

## Inference

### Default sparse inference

```bash
./build/bin/main -m /PATH/TO/MODEL -n 128 -t 8 -p "Once upon a time"
```

Windows:

```powershell
.\build\bin\Release\main.exe -m .\MODEL.powerinfer.gguf -n 128 -t 8 -p "Once upon a time"
```

### Limit GPU memory with `--vram-budget`

```bash
./build/bin/main -m /PATH/TO/MODEL -n 128 -t 8 -p "Once upon a time" --vram-budget 8
```

Under CPU/GPU hybrid inference, PowerInfer offloads dense blocks first and then uses remaining VRAM for sparse FFN placement.

### Dense inference mode

For dense GGUF variants, you can use the familiar layer-offload style:

```bash
./build/bin/main -m /PATH/TO/DENSE/MODEL -n 128 -t 8 -p "Once upon a time" -ngl 12
```

This is a compatibility path, not the primary sparse PowerInfer workflow.

## Serving, evaluation, and other workflows

PowerInfer keeps the same practical local-deployment shape as llama.cpp-style tooling, with PowerInfer-specific sparse behavior where needed.

- [Serving](./examples/server/README.md)
- [Perplexity evaluation](./examples/perplexity/README.md)
- [Batched generation](./examples/batched/README.md)
- [Main CLI usage](./examples/main/README.md)
- [Quantization example](./examples/quantize/README.md)
- [Compression and quantization guidance](./docs/compression-and-quantization.md)
- [Performance troubleshooting](./docs/token_generation_performance_tips.md)

For constrained outputs and agent-style integrations, also look at:

- [Grammar assets](./grammars/README.md)
- [SmallThinker function calling docs](./smallthinker/docs/function-calling.md)
- [SmallThinker install docs](./smallthinker/docs/install.md)
- [SmallThinker multimodal docs](./smallthinker/docs/multimodal.md)

## Server API and agent integration

The built-in server exposes a JSON HTTP API that works directly with local-app and agent workflows. Start it with:

```bash
./build/bin/server \
  -m /PATH/TO/MODEL \
  -c 4096 \
  --port 8080 \
  -np 4 \         # number of parallel request slots
  -cb             # continuous batching for higher throughput
```

### Native completion endpoint

The `/completion` endpoint accepts prompts and returns structured JSON. This is the lowest-latency path for local applications:

```bash
curl -s http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Extract the city name from: \"She moved to Berlin last year.\"",
    "n_predict": 16,
    "temperature": 0.0,
    "stop": ["\n"]
  }' | python3 -m json.tool
```

For streaming token-by-token output, add `"stream": true` and consume the `data:` SSE lines:

```bash
curl -s http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Once upon a time","n_predict":128,"stream":true}'
```

### Structured output and grammar constraints

The server accepts a `grammar` field in any `/completion` request. This is the highest-reliability path for agent-facing endpoints because the model cannot produce tokens that violate the grammar.

JSON-only output:

```bash
curl -s http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Return a JSON object with keys name and city for: John Smith from Paris.",
    "n_predict": 64,
    "temperature": 0.0,
    "grammar": "root   ::= object\nvalue  ::= object | array | string | number | bool | null\nobject ::= \"{\" ws ( string \":\" ws value (\",\" ws string \":\" ws value)* )? \"}\" ws\narray  ::= \"[\" ws ( value (\",\" ws value)* )? \"]\" ws\nstring ::= \"\\\"\" ([^\\\\\"\\x7F\\x00-\\x1F] | \"\\\\\" [\"\\\\/bfnrt] | \"\\\\u\" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])* \"\\\"\" ws\nnumber ::= (\"-\"? ([0-9] | [1-9] [0-9]*)) (\".\" [0-9]+)? (([eE] [-+]? [0-9]+))? ws\nbool   ::= (\"true\" | \"false\") ws\nnull   ::= \"null\" ws\nws     ::= ([ \\t\\n] ws)?"
  }' | python3 -m json.tool
```

Prebuilt grammar files for JSON, lists, arithmetic, and chess are available in [`grammars/`](./grammars/README.md). Pass them from disk with:

```bash
curl -s http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"List three capital cities:\",
    \"n_predict\": 128,
    \"grammar\": $(jq -Rs . < grammars/list.gbnf)
  }"
```

### OpenAI-compatible API wrapper

For integrations that expect the OpenAI SDK surface, run the thin wrapper alongside the server:

```bash
python examples/server/api_like_OAI.py
```

Then point any OpenAI-compatible client at `http://localhost:8081`:

```python
import openai
openai.api_base = "http://localhost:8081/v1"
openai.api_key = "sk-no-key-required"
response = openai.ChatCompletion.create(
    model="powerinfer",
    messages=[{"role": "user", "content": "What is activation sparsity?"}],
)
print(response.choices[0].message.content)
```

### Agent workflow recommendations

| Integration pattern | Recommended approach |
| --- | --- |
| Single-turn extraction, routing, scoring | `/completion` with `grammar`, `temperature: 0`, and a tight `stop` list |
| Multi-turn chat applications | `/completion` with a system prompt loaded via `-spf system.json` and `cache_prompt: true` |
| Streaming UI or typeahead | `/completion` with `stream: true` and incremental SSE handling |
| OpenAI SDK or LangChain integration | `api_like_OAI.py` wrapper |
| Parallel batch requests | Multiple slots with `-np N` and `slot_id` to pin requests |

Key server flags for production-like local deployments:

| Flag | Purpose |
| --- | --- |
| `-np N` | N parallel request slots for concurrent workloads |
| `-cb` | Continuous batching; keeps GPU busy across concurrent slots |
| `--vram-budget N` | Cap VRAM (GiB); useful on shared machines |
| `-c N` | Context window size per slot |
| `--timeout N` | Per-request read/write timeout in seconds |
| `--host 0.0.0.0` | Listen on all interfaces for LAN access |

## Benchmarks and validation

Use this section as the practical checklist when comparing builds, measuring the impact of quantization, or verifying that a model is performing correctly.

### Speed: tokens per second with llama-bench

`llama-bench` measures prompt processing (pp) and text generation (tg) throughput:

```bash
# baseline: prompt processing and generation at default settings
./build/bin/llama-bench \
  -m /PATH/TO/MODEL \
  -p 512 \
  -n 128 \
  -r 3

# compare two quantization levels
./build/bin/llama-bench \
  -m model_fp16.powerinfer.gguf \
  -m model_q4.powerinfer.gguf \
  -p 512 \
  -n 128 \
  -r 3 \
  -o md
```

Output is a markdown table with average t/s and standard deviation. Add `-o json` to capture results for scripted comparison.

Key tuning knobs to vary:

| Flag | What it tests |
| --- | --- |
| `-t N` | Thread count; sweep 1,2,4,8,… to find the CPU throughput cliff |
| `--vram-budget N` | How speed degrades as VRAM budget shrinks |
| `-b N` | Batch size; larger batches improve prompt-processing t/s |
| `-r N` | Repetitions; use at least 3 for stable averages |

### Memory: VRAM and RAM usage

PowerInfer logs VRAM allocation on startup. Check these lines to confirm the hot/cold FFN split is working:

```text
llm_load_sparse_model_tensors: VRAM used: XXXX.XX MB
llm_load_gpu_split: offloaded XXXX.XX MiB of FFN weights to GPU
```

If `VRAM used` is far below your hardware maximum and you did not set `--vram-budget`, see [docs/token_generation_performance_tips.md](./docs/token_generation_performance_tips.md) for FFN split diagnostics.

To measure peak RSS on Linux:

```bash
/usr/bin/time -v ./build/bin/main \
  -m /PATH/TO/MODEL \
  -n 64 \
  -p "Once upon a time" 2>&1 | grep "Maximum resident"
```

### Output quality: perplexity

Perplexity is the standard proxy for output quality. Lower is better; a change of more than ~0.5% versus the baseline is usually meaningful.

```bash
# compute perplexity on a text file (wikitext-2 or any representative document)
./build/bin/perplexity \
  -m /PATH/TO/MODEL \
  -f /PATH/TO/TEST_CORPUS.txt \
  --chunks 10
```

Expected reference ranges from the Llama 2 70B scorechart:

| Quantization | Perplexity | Delta vs fp16 |
| --- | --- | --- |
| fp16 | 3.4313 | — |
| Q6_K | 3.4367 | +0.16 % |
| Q4_K_M | 3.4725 | +1.20 % |
| Q4_0 | 3.5550 | +3.61 % |
| Q2_K | 3.7339 | +8.82 % |

### Validation checklist

Run through this checklist when evaluating a new build, model, or quantization level:

1. **Build check** – `cmake --build build --config Release` completes without errors.
2. **Load check** – `main` starts, logs the sparse model tensors block, and produces output.
3. **FFN split check** – Confirm `llm_load_gpu_split` line appears if running with GPU.
4. **Speed baseline** – `llama-bench` with default settings; record average tg t/s and pp t/s.
5. **Memory baseline** – Note `VRAM used` from startup logs and peak RSS from `/usr/bin/time`.
6. **Quality baseline** – `perplexity --chunks 10`; confirm value is within expected range for the quantization level.
7. **Server smoke test** – Start server, POST to `/completion`, confirm valid JSON response with `content` field.
8. **Grammar smoke test** – POST to `/completion` with a JSON grammar, confirm output parses as valid JSON.

## Hallucination reduction and cognitive load

PowerInfer's practical low-hallucination path is to reduce generation freedom only as much as the task requires, instead of turning every request into a long-reasoning workflow.

### Practical feature set

- **Constrain the format first.** In the root runtime, use `--grammar` / `--grammar-file` in [`main`](./examples/main/README.md) or the `grammar` field in [`server`](./examples/server/README.md). In SmallThinker, prefer `--json-schema`, `json_schema`, or `response_format` for schema-constrained generation.
- **Keep answers bounded.** Use short prompts, conservative `n_predict` limits, stop strings, and low-variance sampling for factual QA, extraction, and classification.
- **Use reasoning only when it earns its cost.** For SmallThinker server workflows, `--reasoning-budget 0` is the default choice for extraction, routing, tool use, and strict structured output; leave reasoning enabled only for tasks that benefit from intermediate deliberation.
- **Prefer machine-readable contracts for machine consumers.** For agents and apps, use schema-first JSON or function calling instead of free-form prose whenever possible.

### Default low-hallucination path

1. Start with **strict JSON or grammar-constrained output**.
2. Keep schemas narrow, with required fields and `"additionalProperties": false` when possible, because tighter schemas produce faster grammars and reduce hallucinated keys.
3. Ask for a **short final answer** when a human is the consumer and formatting flexibility is acceptable.
4. Escalate to **bounded reasoning** only for tasks like multi-step planning, synthesis, or tool-selection decisions that are measurably better with extra thinking.

### Task-oriented guidance

| Task type | Default mode | Why |
| --- | --- | --- |
| Extraction, classification, routing, scoring | Strict JSON / grammar | Highest validity and lowest drift |
| Tool calling and agent handoffs | Function calling or schema-first JSON | Keeps downstream integrations deterministic |
| Factual QA for humans | Short answers with bounded length | Reduces latency and discourages speculative elaboration |
| Harder synthesis or planning | Bounded reasoning, then a short final answer | Gives the model room to work without making long reasoning the default |

### Evaluation tasks to track

- **Hallucination reduction:** compare free-form prompting against constrained-output prompting on extraction, routing, and grounded QA tasks with known answers.
- **JSON validity:** measure parse success, required-field coverage, and schema compliance for representative server and chat-completions payloads.
- **Factual consistency:** test short-answer responses against internal documents or benchmark sets where exact-match or citation agreement can be checked.
- **Latency-quality balance:** track completion latency, output length, and validation pass rate together so low-hallucination defaults stay practical for local inference.

## Quantization

PowerInfer includes optimized support for INT4 (`Q4_0`) quantization:

```bash
./build/bin/quantize /PATH/TO/MODEL /PATH/TO/OUTPUT/QUANTIZED/MODEL Q4_0
```

After quantization, run the quantized model with the same inference commands above.

For a full discussion of which quantization formats work best with PowerInfer's hot/cold split, which tensors benefit most from compression, and which compression approaches are production-ready versus research-only, see [docs/compression-and-quantization.md](./docs/compression-and-quantization.md).

## Fine-tuning addendum

The repository already includes a local adapter fine-tuning path under [`examples/finetune/`](./examples/finetune/README.md).

Use this addendum as the practical guide:

1. **Start from a supported dense base model or dense GGUF workflow.** The built-in fine-tuning flow is adapter-oriented and is documented around GGUF base models and LoRA outputs.
2. **Run `finetune` to train an adapter locally.** See [`examples/finetune/README.md`](./examples/finetune/README.md) for checkpointing, LoRA rank controls, and resume behavior.
3. **Load the adapter during inference.** The generated LoRA adapters can be applied with `main` using `--lora` or `--lora-scaled`.
4. **Optionally export a merged model.** Use [`examples/export-lora/README.md`](./examples/export-lora/README.md) if you want to bake adapters into a derived GGUF.

Important caveats:

- The sparse PowerInfer path is primarily an **inference** runtime.
- If your target deployment is a PowerInfer sparse model, the fine-tuning flow is usually: fine-tune the supported dense/base model or adapter path first, then convert or package for the deployment format you need.
- For newer on-device sparse MoE work, check [`smallthinker/`](./smallthinker/README.md) first, since that is where the repo's current model direction is documented most actively.

## Evaluation

PowerInfer's original release reported up to 11x speedup on Falcon 40B and up to 3x speedup on Llama 2 70B versus llama.cpp on a single RTX 4090.

![github-eval-4090](https://github.com/SJTU-IPADS/PowerInfer/assets/34213478/d700fa6c-77ba-462f-a2fc-3fd21c898f33)

INT4 results on a single RTX 2080 Ti:

![github-eval-2080ti-q4](https://github.com/SJTU-IPADS/PowerInfer/assets/34213478/0fc1bfc4-aafc-4e82-a865-bec0143aff1a)

For more technical detail, see the [PowerInfer paper](https://arxiv.org/abs/2312.12456).

## FAQ

1. **What if I hit `CUDA_ERROR_OUT_OF_MEMORY`?**
   - Rebuild the GPU index with `--reset-gpu-index`.
   - Try a slightly lower `--vram-budget`.
   - Use `--disable-gpu-index` if you need to disable FFN offloading for diagnosis.

2. **Does PowerInfer support every dense model family?**
   - No. The sparse runtime is focused on models with ReLU, ReGLU, or Squared ReLU style activations and related sparse packaging flows.

3. **Where should I start if I care more about phones or edge deployment than desktop sparse inference?**
   - Start with [`smallthinker/README.md`](./smallthinker/README.md).

4. **Where is the active modernization backlog?**
   - See [TODO.md](./TODO.md).

## TODOs

The active modernization tracker lives in [TODO.md](./TODO.md).

## Paper and citation

If you find PowerInfer useful, please cite:

```bibtex
@misc{song2023powerinfer,
      title={PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU},
      author={Yixin Song and Zeyu Mi and Haotong Xie and Haibo Chen},
      year={2023},
      eprint={2312.12456},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgement

We are thankful for [ggml](https://github.com/ggerganov/ggml), [llama.cpp](https://github.com/ggerganov/llama.cpp), the ReLU-based sparse model work from [THUNLP](https://nlp.csai.tsinghua.edu.cn/), and the research direction opened by [Deja Vu](https://proceedings.mlr.press/v202/liu23am.html).
