# Fine-tuning Supported Models

This guide covers how to fine-tune models that are compatible with PowerInfer, including LoRA adapter training for ReLU-sparse base models and SmallThinker MoE models.

## Overview

PowerInfer supports two fine-tuning paths:

| Path | Best for | Tool |
|------|----------|------|
| **LoRA (ReLU-sparse models)** | Bamboo, ProSparse LLaMA-2, ReluLLaMA-2 | `examples/finetune` |
| **HuggingFace → GGUF conversion** | SmallThinker, custom sparse models | `smallthinker/convert_hf_to_gguf.py` |

---

## LoRA Fine-tuning (ReLU-sparse Models)

PowerInfer inherits LoRA fine-tuning support from llama.cpp. The `examples/finetune` binary trains low-rank adapters on top of a GGUF base model entirely on-device.

### Prerequisites

Build the `finetune` binary first:

```bash
cmake -S . -B build
cmake --build build --config Release --target finetune
```

### Basic usage

```bash
# 1. Obtain training data (example: Shakespeare corpus)
wget https://raw.githubusercontent.com/brunoklein99/deep-learning-notes/master/shakespeare.txt

# 2. Train a LoRA adapter
./build/bin/finetune \
    --model-base /PATH/TO/MODEL.gguf \
    --checkpoint-in  chk-lora-MODELNAME-LATEST.gguf \
    --checkpoint-out chk-lora-MODELNAME-ITERATION.gguf \
    --lora-out lora-MODELNAME-ITERATION.bin \
    --train-data "shakespeare.txt" \
    --save-every 10 \
    --threads 6 --adam-iter 30 --batch 4 --ctx 64 \
    --use-checkpointing

# 3. Run inference with the adapter
./build/bin/main -m /PATH/TO/MODEL.gguf --lora lora-MODELNAME-LATEST.bin
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--lora-r N` | 4 | Default LoRA rank (higher = more capacity, more memory) |
| `--adam-iter N` | 256 | Number of training iterations |
| `--batch N` | 8 | Training batch size |
| `--ctx N` | 128 | Training context length |
| `--use-checkpointing` | off | Halves memory use at a small speed cost |
| `--threads N` | 6 | CPU threads for training |
| `--save-every N` | 10 | Checkpoint interval (in iterations) |

### Mixing multiple adapters

You can combine adapters at inference time and control each one's strength:

```bash
./build/bin/main -m /PATH/TO/MODEL.gguf \
  --lora-scaled lora-adapter-a-LATEST.bin 0.7 \
  --lora-scaled lora-adapter-b-LATEST.bin 0.4 \
  --lora lora-adapter-c-LATEST.bin
```

Scale values do not need to sum to 1. Values above 1 amplify the adapter; values below 1 blend it in gently.

### Supported base models for LoRA fine-tuning

- Bamboo-7B (`PowerInfer/Bamboo-base-v0.1-gguf`)
- ProSparse LLaMA-2 7B / 13B (`PowerInfer/prosparse-llama-2-*-gguf`)
- ReluLLaMA-2 7B / 13B / 70B (`PowerInfer/ReluLLaMA-*-PowerInfer-GGUF`)
- Any dense GGUF model (dense inference mode, no sparsity gain)

> **Note:** Fine-tuning changes the adapter weights only, not the base model. The resulting adapter is fully compatible with the `main` and `server` examples.

---

## SmallThinker: Converting Fine-tuned HuggingFace Checkpoints

SmallThinker models use native Mixture-of-Experts (MoE) architecture and must be fine-tuned at the HuggingFace level, then converted to GGUF for PowerInfer deployment.

### Step 1 — Fine-tune in HuggingFace format

Use a standard HuggingFace trainer (e.g. `transformers`, `trl`, `unsloth`) targeting the SmallThinker base weights:

- [SmallThinker-21BA3B-Instruct](https://huggingface.co/PowerInfer/SmallThinker-21BA3B-Instruct)
- [SmallThinker-4BA0.6B-Instruct](https://huggingface.co/PowerInfer/SmallThinker-4BA0.6B-Instruct)

Save your checkpoint as a standard HuggingFace safetensors directory.

### Step 2 — Convert to GGUF

Run the conversion script from the `smallthinker/` directory:

```bash
cd smallthinker

# Convert to FP16 GGUF
python3 convert_hf_to_gguf.py /path/to/your-finetuned-checkpoint \
    --outtype f16 \
    --outfile /path/to/output-model.gguf \
    --transpose-down all
```

### Step 3 — Quantize (optional)

```bash
./build/bin/llama-quantize --pure \
    /path/to/output-model.gguf \
    /path/to/output-model-q4_0.gguf \
    Q4_0 8
```

### Step 4 — Run inference

```bash
cd smallthinker
./build/bin/llama-cli -m /path/to/output-model-q4_0.gguf -p "Your prompt here"
```

See the full [SmallThinker README](../smallthinker/README.md) for build instructions, supported hardware, and benchmark results.

---

## Converting Custom HuggingFace ReLU Models

If you have a custom HuggingFace model with a ReLU/ReGLU/Squared-ReLU activation function, convert it to PowerInfer GGUF format using `convert-hf-to-powerinfer-gguf.py`:

```bash
pip install -r requirements.txt

python convert-hf-to-powerinfer-gguf.py \
    --outfile /PATH/TO/OUTPUT.powerinfer.gguf \
    /PATH/TO/HF_MODEL \
    /PATH/TO/PREDICTOR_WEIGHTS   # optional: predictor for hot-neuron offloading
```

Without a predictor the model runs in dense mode (no GPU-sparse offloading benefit). Predictor weights are available on Hugging Face for the officially supported model families — see the [model weights table](../README.md#model-weights).

---

## Further Reading

- [LoRA fine-tuning reference (`examples/finetune`)](../examples/finetune/README.md)
- [SmallThinker on-device inference](../smallthinker/README.md)
- [Performance troubleshooting](./token_generation_performance_tips.md)
- [PowerInfer paper](https://ipads.se.sjtu.edu.cn/_media/publications/powerinfer-20231219.pdf)
