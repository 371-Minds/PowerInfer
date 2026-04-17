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
| [`docs/`](./docs) | Root-project troubleshooting and backend notes |
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

- [Build the root project](#build)
- [Get model weights](#model-weights)
- [Run inference](#inference)
- [Serve or benchmark a model](#serving-evaluation-and-other-workflows)
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
- [Performance troubleshooting](./docs/token_generation_performance_tips.md)

For constrained outputs and agent-style integrations, also look at:

- [Grammar assets](./grammars/README.md)
- [SmallThinker function calling docs](./smallthinker/docs/function-calling.md)
- [SmallThinker install docs](./smallthinker/docs/install.md)
- [SmallThinker multimodal docs](./smallthinker/docs/multimodal.md)

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
