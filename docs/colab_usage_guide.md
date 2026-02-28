# 📘 UpFlame-AGO Colab Usage Guide

This guide explains how to properly set up, train, and test the UpFlame-AGO repository within a Google Colab environment (optimized for the Free Tier T4 GPU, ~15GB VRAM).

By following this guide, you will learn how to bypass the heavier orchestrator/RL components to establish a clean, scalable Transformer baseline, progressively scaling from a 100M model up to validating a 2B parameter architecture.

---

## 1. Project Architecture Overview

UpFlame-AGO is built with a highly modular, MNC-grade (Multinational Corporation) structure:
*   **`model/`**: Contains the core `UnifiedTransformer` and `UpFlameAGOUnifiedConfig`. It includes advanced modules (MoE, Memory, World State) which we will toggle off for initial Colab training.
*   **`configs/`**: Stores scaling presets (100M to 2B) and hyperparameter definitions.
*   **`training/`**: Contains our entry points for pre-training and scaling.
*   **`inference/`**: Scripts for text generation from saved checkpoints.
*   **`agent/`, `orchestrator/`, `memory/`, `world_model/`**: Advanced Phase 2 components. **Do not activate these during Week 1 Colab baseline training.**

---

## 2. Colab Environment Setup

Open a new Google Colab notebook and ensure your runtime is set to GPU (T4).

### Step 1: Clone the Repository
Clone the repository directly into the Colab environment.

```python
!git clone https://github.com/kitretsusaisama/Upflame-AGO.git
%cd Upflame-AGO
```

### Step 2: Safe Dependency Installation
UpFlame-AGO uses `pyproject.toml`. However, to prevent Colab environment conflicts, use this granular installation method:

```python
import importlib

def ensure_package(pkg):
    try:
        importlib.import_module(pkg)
    except ImportError:
        print(f"Installing {pkg}...")
        !pip install -q {pkg}

ensure_package("torch")
ensure_package("transformers")
ensure_package("datasets")
ensure_package("sentencepiece")
ensure_package("accelerate")

# Optional package for 8-bit optimization
try:
    import bitsandbytes
except ImportError:
    print("bitsandbytes not installed. Training will fallback to standard torch AdamW.")
    # Uncomment the line below to install if you are on a GPU runtime
    # !pip install -q bitsandbytes
```

### Step 3: Verify Hardware
Confirm your hardware is accessible. This logic correctly identifies TPUs or GPUs and falls back to CPU natively.

```python
import torch

def get_device():
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"✅ TPU Detected: {device}")
        return "tpu"
    except ImportError:
        pass

    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        print("⚠️ No GPU detected. CPU fallback mode will be engaged.")
        return "cpu"

get_device()
```

---

## 3. Disabling Advanced Modules (Colab Safe Mode)

For stable Colab training, we must disable heavy architectural features like Mixture of Experts (MoE) and Infini-attention.

**Ensure your `configs/scaling.yaml` or run command explicitly sets `use_moe=False` and `use_infini_attention=False`.** The `training/train_small.py` script automatically manages this for Week 1 baselines.

---

## 4. Progressive Training Execution

We provide a lightweight training entry point, `training/train_small.py`, designed specifically for this progressive scaling roadmap.

### Day 1: Train 100M Baseline
Run a quick test to ensure the data pipeline and model compile correctly.

```bash
!python training/train_small.py --scale 100M --use_wandb
```
*Outputs checkpoint to: `checkpoints/100M/`*

### Day 2: Scale to 300M
Increase capacity. The script will automatically handle gradient accumulation.

```bash
!python training/train_small.py --scale 300M --use_wandb
```

### Day 3-4: Scale to 700M
At this scale, gradient checkpointing and mixed precision (bf16/fp16) become strictly necessary.

```bash
!python training/train_small.py --scale 700M --use_wandb
```

### Day 5: Scale to 1B
This is near the limit of the T4 GPU. Training will be slower but should remain stable.

```bash
!python training/train_small.py --scale 1B --use_wandb
```

---

## 5. Validating 2B Architecture

You cannot perform full training on a 2B model using a standard 15GB T4 GPU. However, you must validate that the architecture compiles and passes a forward pass without OOM (Out of Memory) errors.

Run the validation command:

```bash
!python training/train_small.py --scale 2B --validate_only
```
*This command initializes the model, runs a single dummy batch, and deletes the model to free VRAM.*

---

## 6. Running Inference & Evaluation

To test your trained models, use the dedicated inference script. Point it to the directory of your saved checkpoint.

**For Text Generation:**
```bash
# Example: Running inference on the 300M model
!python inference/run_inference.py --checkpoint checkpoints/300M --prompt "The future of open source AI is"
```

**For MNC-Grade Precise Evaluation:**
```bash
# Example: Running exact perplexity calculation on a prompt
!python inference/run_inference.py --checkpoint checkpoints/300M --prompt "The future of open source AI is" --evaluate
```

---

## 7. CPU-Only Fallback Mode

If you run out of Colab compute units and drop to a CPU runtime, the system can gracefully adapt without crashing.

Append the `--cpu_mode` flag to your command:

```bash
!python training/train_small.py --scale 100M --cpu_mode
```
**What this does internally:**
*   Forces the model to a smaller context window (512 tokens).
*   Reduces batch size.
*   Disables mixed precision (`torch.cuda.amp.autocast`).
*   Falls back to standard `torch.optim.AdamW`.

---

## 8. Analyzing Results

To visualize your scaling laws (Loss vs. Steps / Model Size):

```bash
!python training/plot_scaling.py --log_dir logs/
```
*This script reads the JSON logs generated during training and outputs a matplotlib graph.* *(Note: We also natively integrate with Weights & Biases via the `--use_wandb` flag).*

---

## 9. Common Colab Errors & Fixes

| Error | Cause | Fix |
| :--- | :--- | :--- |
| **CUDA Out of Memory (OOM)** | Batch size or context length too high for model scale. | Ensure `--scale` is 1B or lower. Ensure `use_moe=False`. Restart runtime. |
| **bitsandbytes initialization failed** | Mismatched CUDA versions in Colab. | The script will automatically fallback to `torch.optim.AdamW`. You can ignore the warning. |
| **Runtime Disconnected** | Colab timeout or RAM limit reached. | Ensure you are streaming datasets (like `wikitext-2`). Reconnect and run with `--resume_from_checkpoint`. |
| **Pickle / State Dict mismatch** | Loading a GPU checkpoint on CPU. | Fixed internally via `map_location=device` in `run_inference.py`. |

---

## 10. Suggested Refactoring Checklist (Pre-Colab)

Before starting, ensure your repository has the following files (or create them based on the provided specifications):
1.  `training/train_small.py`: The lightweight CLI entry point.
2.  `configs/scaling.yaml`: Defines layer, hidden dimension, and head counts for 100M -> 2B.
3.  `inference/run_inference.py`: Standalone generation script.
4.  `training/plot_scaling.py`: Visualization utility.

*(Refer to the implementation details of these scripts for full code integration).*
