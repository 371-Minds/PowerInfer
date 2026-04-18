# AGENTS.md – PowerInfer repository guide for AI coding agents

This file describes how to navigate, build, test, and contribute to the PowerInfer repository. It is written for AI coding agents (Copilot, Claude, Cursor, Codex, etc.) and human contributors who want a fast orientation.

---

## What this repository is

PowerInfer is a C++-based local LLM inference engine that exploits **activation locality** to run sparse models efficiently on consumer hardware. The project has two main deployment tracks:

- **PowerInfer classic** – hybrid CPU/GPU inference for ReLU-sparse GGUF models. Code lives in the root of the repository.
- **SmallThinker** – newer on-device sparse MoE runtime for phones and edge devices. Code lives under `smallthinker/`.

The project is closely related to [llama.cpp](https://github.com/ggerganov/llama.cpp); many conventions, tooling patterns, and example names are inherited from it.

---

## Repository layout

```
PowerInfer/
├── ggml.c / ggml.h          Core tensor library (CPU)
├── ggml-cuda.cu / .h        CUDA backend
├── ggml-metal.m / .h        Metal backend (macOS)
├── ggml-opencl.cpp / .h     OpenCL backend
├── ggml-backend.*           Backend abstraction layer
├── llama.cpp / llama.h      Main model loading, sampling, KV cache
├── common/                  Shared CLI argument parsing, sampling helpers
├── examples/                CLI tools and usage examples
│   ├── main/                CLI inference (the primary end-user binary)
│   ├── server/              HTTP API server (JSON + SSE)
│   ├── llama-bench/         Performance benchmarking tool
│   ├── perplexity/          Output quality measurement
│   ├── batched/             Batched generation example
│   ├── quantize/            Model quantization tool
│   ├── finetune/            LoRA fine-tuning
│   └── export-lora/         Merge LoRA adapters into GGUF
├── docs/                    Root-project documentation
│   ├── compression-and-quantization.md
│   └── token_generation_performance_tips.md
├── grammars/                GBNF grammar files for constrained generation
├── gguf-py/                 Python GGUF file tooling
├── powerinfer-py/           Python helpers for PowerInfer workflows
├── smallthinker/            SmallThinker on-device MoE runtime (separate subtree)
├── convert.py               Convert original + predictor weights → PowerInfer GGUF
├── convert-dense.py         Convert original weights → dense GGUF
├── convert-hf-to-powerinfer-gguf.py  HuggingFace → PowerInfer GGUF conversion
├── CMakeLists.txt           Root CMake build file
├── TODO.md                  Active modernization backlog
└── README.md                Project overview and user guide
```

---

## Build

All build commands run from the repository root. The build output lands in `build/bin/`.

### CPU-only (no GPU dependencies required)

```bash
cmake -S . -B build
cmake --build build --config Release -j$(nproc)
```

### NVIDIA GPU (CUDA)

```bash
cmake -S . -B build -DLLAMA_CUBLAS=ON
cmake --build build --config Release -j$(nproc)
```

### AMD GPU (ROCm/HIP)

```bash
# Replace gfx1100 with your GPU architecture from `rocminfo`
CC=/opt/rocm/llvm/bin/clang \
CXX=/opt/rocm/llvm/bin/clang++ \
cmake -S . -B build -DLLAMA_HIPBLAS=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build --config Release -j$(nproc)
```

### Key CMake flags

| Flag | Purpose |
| --- | --- |
| `-DLLAMA_CUBLAS=ON` | Enable CUDA backend |
| `-DLLAMA_HIPBLAS=ON` | Enable ROCm/HIP backend |
| `-DLLAMA_METAL=ON` | Enable Metal backend (macOS) |
| `-DLLAMA_BLAS=ON` | Enable BLAS for CPU path |
| `-DLLAMA_NATIVE=ON` | Use `-march=native` for current CPU |
| `-DCMAKE_BUILD_TYPE=Debug` | Debug build (slower, with symbols) |

---

## Tests

The test suite is driven by CMake/CTest.

```bash
# run all tests
cd build && ctest --output-on-failure

# run a specific test by name pattern
ctest -R test-tokenizer --output-on-failure

# run tests with verbose output
ctest -V
```

Individual test binaries are also in `build/bin/` and can be run directly for faster iteration.

---

## Python tooling

Install Python dependencies before using conversion or evaluation scripts:

```bash
pip install -r requirements.txt
```

Key Python entry points:

| Script | Purpose |
| --- | --- |
| `convert.py` | Convert base model + predictor weights → PowerInfer GGUF |
| `convert-dense.py` | Convert base model → dense GGUF (no sparse routing) |
| `convert-hf-to-powerinfer-gguf.py` | HuggingFace format → PowerInfer GGUF |
| `gguf-py/` | Python package for reading and writing GGUF files |
| `powerinfer-py/` | Python helpers that the runtime invokes for GPU index generation |

---

## Key concepts for making changes

### Activation locality and sparse routing

PowerInfer's core claim is that during LLM inference, a subset of FFN neurons ("hot neurons") are consistently activated across many inputs. The `convert.py` flow bundles a learned **predictor** alongside the model weights. At runtime, the predictor decides which neurons to run on GPU and which to run on CPU, enabling hybrid execution with lower VRAM than a fully offloaded dense model.

### PowerInfer GGUF format

PowerInfer uses an extended GGUF format (`*.powerinfer.gguf`) that embeds predictor tensors and activation statistics. Standard GGUF tools may not fully understand these extensions. When modifying tensor handling code, check `llama.cpp` for the `gpu_idx` tensor logic and the `generated.gpuidx` file handling.

### Hot/cold split and VRAM budget

`--vram-budget N` (GiB) controls how much VRAM PowerInfer allocates. Dense layers (attention, layer norms) are offloaded first; remaining VRAM is used for hot FFN neuron placement. If VRAM is exhausted, those neurons fall back to CPU. The GPU index is generated once and cached as a `.generated.gpuidx` file alongside the model.

### Grammar-constrained generation

The root runtime uses [GBNF grammars](./grammars/README.md) via `--grammar` / `--grammar-file` in `main` and the `grammar` field in server `/completion` requests. Grammar files live in `grammars/`. When adding new grammar support or testing constrained generation, use `perplexity` or `main` with a small grammar file.

---

## Making changes

### Where things live for common tasks

| Task | Files to look at |
| --- | --- |
| Model loading and KV cache | `llama.cpp`, `llama.h` |
| Tensor math and backends | `ggml.c`, `ggml-cuda.cu`, `ggml-metal.m` |
| CLI argument parsing | `common/common.cpp`, `common/common.h` |
| Server HTTP API | `examples/server/server.cpp` |
| Sampling logic | `common/sampling.cpp`, `common/sampling.h` |
| GGUF read/write | `gguf-py/`, `llama.cpp` GGUF section |
| GPU index generation | `powerinfer-py/` |
| Model conversion | `convert.py`, `convert-hf-to-powerinfer-gguf.py` |

### Code style

- C and C++ files follow the style of the surrounding code (llama.cpp lineage: K&R-adjacent, `snake_case` for functions and variables, `UPPER_CASE` for macros).
- Python files use `snake_case` and should pass `flake8` (config in `.flake8`).
- Do not add comments that merely restate what the code does; comments should explain *why*.
- Prefer small, focused diffs. PowerInfer inherits changes from llama.cpp upstream; large structural changes create merge conflicts.

### Pre-commit hooks

The repository includes a `.pre-commit-config.yaml`. To activate:

```bash
pip install pre-commit
pre-commit install
```

Hooks include `clang-format` for C/C++ and `flake8` for Python.

---

## Smoke-test an end-to-end change

After modifying inference code, the minimum verification sequence is:

```bash
# 1. rebuild
cmake --build build --config Release -j$(nproc)

# 2. run ctest
cd build && ctest --output-on-failure && cd ..

# 3. run a short inference
./build/bin/main \
  -m /PATH/TO/MODEL \
  -n 32 \
  -t 4 \
  -p "The capital of France is"

# 4. run a server smoke test
./build/bin/server -m /PATH/TO/MODEL -c 512 --port 8080 &
sleep 2
curl -s http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","n_predict":8}' | python3 -m json.tool
kill %1
```

---

## Documentation

- `README.md` – top-level user guide; update this when adding user-facing features or changing CLI flags.
- `TODO.md` – active modernization backlog; mark items done when completing planned work.
- `AGENTS.md` – this file; update when the build, test, or repository layout changes.
- `docs/` – deeper technical documentation for root-project topics.
- `smallthinker/docs/` – SmallThinker-specific documentation.
- `examples/*/README.md` – per-example documentation; keep in sync with option changes.

---

## Relationship to llama.cpp

PowerInfer is a fork and extension of llama.cpp. The core tensor library (`ggml`), the model loading layer (`llama.cpp`/`llama.h`), and the example structure all originate from llama.cpp. PowerInfer adds:

- Sparse FFN routing via learned predictors
- GPU index generation and the `.powerinfer.gguf` format
- The `--vram-budget` control flow
- SmallThinker models and the on-device MoE runtime

When merging upstream llama.cpp changes, pay close attention to `llama.cpp` (model loading), `ggml.c`, and the GGUF handling sections, as these are most likely to conflict with PowerInfer extensions.
