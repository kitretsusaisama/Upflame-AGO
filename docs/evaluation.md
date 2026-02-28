# UpFlame-AGO Evaluation Guide

This document outlines how to evaluate the UpFlame-AGO foundation model using standard evaluation metrics.

## Setup

First, ensure you have the required dependencies for evaluation. You can install standard frameworks like `lm-eval` (Language Model Evaluation Harness):

```bash
pip install lm-eval
```

## Running Evaluation

Currently, testing generation and manual evaluation can be done using the `inference/run_inference.py` script:

```bash
python inference/run_inference.py \
  --checkpoint checkpoints/100M \
  --prompt "The future of open source AI is" \
  --max_new_tokens 50 \
  --temperature 0.7
```

### Precise Prompt Perplexity Evaluation

To precisely evaluate the exact perplexity and cross-entropy loss of the model on a given input prompt (MNC-grade self-evaluation), append the `--evaluate` flag:

```bash
python inference/run_inference.py \
  --checkpoint checkpoints/100M \
  --prompt "The future of open source AI is a collaborative and transparent ecosystem." \
  --evaluate
```

## Full Model Evaluation via Harness

For automated metrics on language tasks (e.g. WikiText, Lambada) or coding tasks (e.g. HumanEval), you can run:

```bash
# Evaluate on language tasks
lm_eval --model hf --model_args pretrained=checkpoints/100M --tasks wikitext,lambada_openai --device cuda

# Evaluate on coding benchmarks (HumanEval)
lm_eval --model hf --model_args pretrained=checkpoints/100M --tasks humaneval --device cuda
```

*Note: You may need to export your checkpoint to a standard HuggingFace format using a conversion script prior to running `lm-eval` with the `--model hf` flag if the baseline model is not recognized immediately.*
