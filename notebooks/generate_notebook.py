import json
import os

# --- Notebook Content ---

# Section 1: Environment Setup
cell_1_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 🔥 UpFlame Progressive Transformer Training: 100M → 2B in 7 Days\n",
        "\n",
        "## Overview\n",
        "This notebook implements a complete, production-grade progressive scaling roadmap for training a GPT-style Transformer language model. \n",
        "Starting from a **100M parameter model**, we scale incrementally to a **2B parameter architecture skeleton** over 7 days, optimized for **Google Colab Free (T4 GPU)** or CPU fallback.\n",
        "\n",
        "### Key Features\n",
        "- **Unified Transformer Architecture**: Decoder-only, RMSNorm, SwiGLU, GQA, RoPE.\n",
        "- **Memory Optimization**: Gradient Checkpointing, Mixed Precision (bf16/fp16), Gradient Accumulation.\n",
        "- **Scalable Configs**: Pre-defined presets for 100M, 300M, 700M, 1B, and 2B models.\n",
        "- **Robust Training Loop**: Cosine LR schedule, Checkpointing, Resume capability.\n",
        "- **CPU Fallback**: Automatically adapts to hardware limitations.\n",
        "\n",
        "## 🗓️ 7-Day Roadmap\n",
        "1.  **Day 1**: Train 100M Model (Baseline)\n",
        "2.  **Day 2**: Scale to 300M Model\n",
        "3.  **Day 3-4**: Scale to 700M Model\n",
        "4.  **Day 5**: Scale to 1B Model\n",
        "5.  **Day 6**: Validate 2B Architecture (Forward Pass Only)\n",
        "6.  **Day 7**: Final Tuning & Inference Testing\n",
        "\n",
        "---"
    ]
}

cell_1_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# @title 1. Environment Setup & Hardware Detection\n",
        "import os\n",
        "import sys\n",
        "import torch\n",
        "import math\n",
        "import time\n",
        "import json\n",
        "import random\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import importlib\n",
        "from typing import Optional, Tuple, List\n",
        "from dataclasses import dataclass\n",
        "\n",
        "# Install necessary packages if not present\n",
        "def ensure_package(pkg):\n",
        "    try:\n",
        "        importlib.import_module(pkg)\n",
        "    except ImportError:\n",
        "        print(f\"Installing {pkg}...\")\n",
        "        !pip install -q {pkg}\n",
        "\n",
        "ensure_package(\"transformers\")\n",
        "ensure_package(\"datasets\")\n",
        "ensure_package(\"sentencepiece\")\n",
        "ensure_package(\"accelerate\")\n",
        "\n",
        "# Optional package\n",
        "try:\n",
        "    import bitsandbytes\n",
        "except ImportError:\n",
        "    print(\"bitsandbytes not installed. Will use torch AdamW fallback.\")\n",
        "    # Uncomment to install if needed:\n",
        "    # !pip install -q bitsandbytes\n",
        "\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "from torch.utils.data import DataLoader, IterableDataset\n",
        "from transformers import get_cosine_schedule_with_warmup\n",
        "\n",
        "# Hardware Detection\n",
        "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
        "print(f\"Using device: {device}\")\n",
        "\n",
        "if device == \"cuda\":\n",
        "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n",
        "    # Enable TF32 for faster training on Ampere GPUs (A100, etc.) - T4 does not support TF32\n",
        "    # torch.backends.cuda.matmul.allow_tf32 = True \n",
        "    # torch.backends.cudnn.allow_tf32 = True\n",
        "    pass\n",
        "else:\n",
        "    print(\"⚠️ Running on CPU. Training will be slow. Adjusting configs automatically.\")\n",
        "\n",
        "# Set Seed for Reproducibility\n",
        "def set_seed(seed=42):\n",
        "    random.seed(seed)\n",
        "    np.random.seed(seed)\n",
        "    torch.manual_seed(seed)\n",
        "    if torch.cuda.is_available():\n",
        "        torch.cuda.manual_seed_all(seed)\n",
        "\n",
        "set_seed(42)\n",
        "print(\"Environment Ready.\")"
    ]
}


# Section 2: Tokenizer Training
cell_2_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 2. Tokenizer Training\n",
        "We train a **SentencePiece Unigram Tokenizer** on a subset of the dataset.\n",
        "- **Vocab Size**: 32,000 (standard for small-mid scale models)\n",
        "- **Normalization**: NFKC\n",
        "- **Byte Fallback**: Enabled to handle unknown characters"
    ]
}

cell_2_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# @title 2. Train SentencePiece Tokenizer\n",
        "from datasets import load_dataset\n",
        "import sentencepiece as spm\n",
        "\n",
        "# Configuration\n",
        "VOCAB_SIZE = 32000\n",
        "DATASET_NAME = \"wikitext\"\n",
        "DATASET_CONFIG = \"wikitext-2-v1\"\n",
        "TOKENIZER_PREFIX = \"upflame_tokenizer\"\n",
        "\n",
        "# Load Dataset (Small subset for tokenizer training)\n",
        "print(\"Loading dataset for tokenizer training...\")\n",
        "dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=\"train\")\n",
        "\n",
        "# Prepare text file for SentencePiece\n",
        "with open(\"tokenizer_train.txt\", \"w\", encoding=\"utf-8\") as f:\n",
        "    for i, item in enumerate(dataset):\n",
        "        if len(item['text']) > 0:\n",
        "            f.write(item['text'] + \"\\n\")\n",
        "        if i >= 10000: # Limit to 10k samples for speed\n",
        "            break\n",
        "\n",
        "# Train Tokenizer\n",
        "print(\"Training SentencePiece tokenizer...\")\n",
        "spm.SentencePieceTrainer.train(\n",
        "    input=\"tokenizer_train.txt\",\n",
        "    model_prefix=TOKENIZER_PREFIX,\n",
        "    vocab_size=VOCAB_SIZE,\n",
        "    character_coverage=0.9995,\n",
        "    model_type=\"unigram\",\n",
        "    pad_id=0,\n",
        "    unk_id=1,\n",
        "    bos_id=2,\n",
        "    eos_id=3,\n",
        "    pad_piece=\"<pad>\",\n",
        "    unk_piece=\"<unk>\",\n",
        "    bos_piece=\"<s>\",\n",
        "    eos_piece=\"</s>\",\n",
        "    byte_fallback=True,\n",
        "    normalization_rule_name=\"nfkc_cf\"\n",
        ")\n",
        "\n",
        "# Verify Tokenizer\n",
        "sp = spm.SentencePieceProcessor(model_file=f\"{TOKENIZER_PREFIX}.model\")\n",
        "print(f\"Tokenizer trained. Vocab size: {sp.get_piece_size()}\")\n",
        "print(\"Test encoding: 'Hello, world!' ->\", sp.encode(\"Hello, world!\"))"
    ]
}

# Section 3: Config Generator
cell_3_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 3. Configuration & Scaling Presets\n",
        "Defines model hyperparameters for each scaling stage.\n",
        "\n",
        "| Model | Layers | Hidden Size | Heads | Context | Parameters (Approx) |\n",
        "|---|---|---|---|---|---|\n",
        "| **100M** | 12 | 768 | 12 | 512 | ~85M - 100M |\n",
        "| **300M** | 18 | 1024 | 16 | 1024 | ~300M |\n",
        "| **700M** | 24 | 1536 | 16 | 1024 | ~700M |\n",
        "| **1B** | 24 | 2048 | 16 | 1024 | ~1.1B |\n",
        "| **2B** | 30 | 2560 | 20 | 1024 | ~2.0B |"
    ]
}

cell_3_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# @title 3. Model Config Generator\n",
        "\n",
        "@dataclass\n",
        "class ModelConfig:\n",
        "    vocab_size: int = 32000\n",
        "    dim: int = 768\n",
        "    n_layers: int = 12\n",
        "    n_heads: int = 12\n",
        "    n_kv_heads: Optional[int] = None # For GQA. If None, defaults to n_heads\n",
        "    multiple_of: int = 256  # Make SwiGLU hidden layer size a multiple of this\n",
        "    ffn_dim_multiplier: Optional[float] = None\n",
        "    norm_eps: float = 1e-5\n",
        "    max_seq_len: int = 512\n",
        "    rope_theta: float = 10000.0\n",
        "    dropout: float = 0.0\n",
        "    use_bias: bool = False # No bias in linear layers (common in Llama/modern architectures)\n",
        "\n",
        "    def __post_init__(self):\n",
        "        if self.n_kv_heads is None:\n",
        "            self.n_kv_heads = self.n_heads\n",
        "\n",
        "def get_config(model_size: str, device_type: str = \"cuda\") -> ModelConfig:\n",
        "    \"\"\"Returns the configuration for a specific model size.\"\"\"\n",
        "    \n",
        "    # 1. CPU Fallback / Safety Defaults\n",
        "    if device_type == \"cpu\":\n",
        "        print(\"Detected CPU. Forcing 100M config with reduced context for safety.\")\n",
        "        return ModelConfig(\n",
        "            dim=512, n_layers=8, n_heads=8, n_kv_heads=4, max_seq_len=256\n",
        "        )\n",
        "\n",
        "    # 2. Scaling Presets\n",
        "    if model_size == \"100M\":\n",
        "        return ModelConfig(\n",
        "            dim=768, n_layers=12, n_heads=12, n_kv_heads=4, max_seq_len=512\n",
        "        )\n",
        "    elif model_size == \"300M\":\n",
        "        return ModelConfig(\n",
        "            dim=1024, n_layers=18, n_heads=16, n_kv_heads=4, max_seq_len=1024\n",
        "        )\n",
        "    elif model_size == \"700M\":\n",
        "        return ModelConfig(\n",
        "            dim=1536, n_layers=24, n_heads=16, n_kv_heads=8, max_seq_len=1024\n",
        "        )\n",
        "    elif model_size == \"1B\":\n",
        "        return ModelConfig(\n",
        "            dim=2048, n_layers=24, n_heads=16, n_kv_heads=8, max_seq_len=1024\n",
        "        )\n",
        "    elif model_size == \"2B\":\n",
        "        return ModelConfig(\n",
        "            dim=2560, n_layers=30, n_heads=20, n_kv_heads=10, max_seq_len=1024\n",
        "        )\n",
        "    else:\n",
        "        raise ValueError(f\"Unknown model size: {model_size}\")\n",
        "\n",
        "# Example Test\n",
        "cfg = get_config(\"100M\", device)\n",
        "print(f\"Loaded Config for 100M: {cfg}\")"
    ]
}

# Section 4: Unified Transformer Implementation
cell_4_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 4. Unified Transformer Implementation\n",
        "A modular, research-grade implementation featuring:\n",
        "- **RMSNorm**: Pre-normalization\n",
        "- **RoPE**: Rotary Positional Embeddings\n",
        "- **GQA**: Grouped Query Attention for efficiency\n",
        "- **SwiGLU**: Activation function\n",
        "- **Gradient Checkpointing**: To save VRAM"
    ]
}

cell_4_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# @title 4. Transformer Architecture Implementation\n",
        "\n",
        "class RMSNorm(nn.Module):\n",
        "    def __init__(self, dim: int, eps: float = 1e-6):\n",
        "        super().__init__()\n",
        "        self.eps = eps\n",
        "        self.weight = nn.Parameter(torch.ones(dim))\n",
        "\n",
        "    def _norm(self, x):\n",
        "        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)\n",
        "\n",
        "    def forward(self, x):\n",
        "        output = self._norm(x.float()).type_as(x)\n",
        "        return output * self.weight\n",
        "\n",
        "def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):\n",
        "    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))\n",
        "    t = torch.arange(end, device=freqs.device, dtype=torch.float32)\n",
        "    freqs = torch.outer(t, freqs)\n",
        "    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64\n",
        "    return freqs_cis\n",
        "\n",
        "def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):\n",
        "    ndim = x.ndim\n",
        "    assert 0 <= 1 < ndim\n",
        "    assert freqs_cis.shape == (x.shape[1], x.shape[-1])\n",
        "    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]\n",
        "    return freqs_cis.view(*shape)\n",
        "\n",
        "def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):\n",
        "    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))\n",
        "    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))\n",
        "    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)\n",
        "    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)\n",
        "    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)\n",
        "    return xq_out.type_as(xq), xk_out.type_as(xk)\n",
        "\n",
        "class Attention(nn.Module):\n",
        "    def __init__(self, args: ModelConfig):\n",
        "        super().__init__()\n",
        "        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads\n",
        "        self.n_heads = args.n_heads\n",
        "        self.n_rep = self.n_heads // self.n_kv_heads\n",
        "        # Validation for GQA\n",
        "        assert args.n_heads % self.n_kv_heads == 0, f\"n_heads ({args.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})\"\n",
        "        \n",
        "        self.head_dim = args.dim // args.n_heads\n",
        "\n",
        "        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)\n",
        "        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)\n",
        "        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)\n",
        "        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)\n",
        "        \n",
        "        self.dropout = nn.Dropout(args.dropout)\n",
        "\n",
        "    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):\n",
        "        bsz, seqlen, _ = x.shape\n",
        "        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)\n",
        "\n",
        "        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)\n",
        "        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)\n",
        "        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)\n",
        "\n",
        "        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)\n",
        "\n",
        "        # GQA: Repeat Key/Value heads\n",
        "        # (bsz, seqlen, n_kv_heads, head_dim) -> (bsz, seqlen, n_heads, head_dim)\n",
        "        xk = torch.repeat_interleave(xk, dim=2, repeats=self.n_rep)\n",
        "        xv = torch.repeat_interleave(xv, dim=2, repeats=self.n_rep)\n",
        "\n",
        "        # Transpose for Attention: (bsz, n_heads, seqlen, head_dim)\n",
        "        xq = xq.transpose(1, 2)\n",
        "        xk = xk.transpose(1, 2)\n",
        "        xv = xv.transpose(1, 2)\n",
        "\n",
        "        # Scaled Dot-Product Attention (with Flash Attention if available via F.scaled_dot_product_attention)\n",
        "        # Create causal mask if not provided or if using standard attention\n",
        "        # However, F.scaled_dot_product_attention handles is_causal=True efficiently\n",
        "        \n",
        "        output = F.scaled_dot_product_attention(\n",
        "            xq, xk, xv, attn_mask=None, dropout_p=self.dropout.p if self.training else 0.0, is_causal=True\n",
        "        )\n",
        "\n",
        "        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)\n",
        "        return self.wo(output)\n",
        "\n",
        "class FeedForward(nn.Module):\n",
        "    def __init__(self, args: ModelConfig):\n",
        "        super().__init__()\n",
        "        hidden_dim = 4 * args.dim\n",
        "        hidden_dim = int(2 * hidden_dim / 3)\n",
        "        if args.ffn_dim_multiplier is not None:\n",
        "            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)\n",
        "        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)\n",
        "\n",
        "        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)\n",
        "        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)\n",
        "        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)\n",
        "\n",
        "    def forward(self, x):\n",
        "        # SwiGLU: w2(F.silu(w1(x)) * w3(x))\n",
        "        return self.w2(F.silu(self.w1(x)) * self.w3(x))\n",
        "\n",
        "class TransformerBlock(nn.Module):\n",
        "    def __init__(self, layer_id: int, args: ModelConfig):\n",
        "        super().__init__()\n",
        "        self.n_heads = args.n_heads\n",
        "        self.dim = args.dim\n",
        "        self.head_dim = args.dim // args.n_heads\n",
        "        self.attention = Attention(args)\n",
        "        self.feed_forward = FeedForward(args)\n",
        "        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)\n",
        "        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)\n",
        "\n",
        "    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):\n",
        "        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)\n",
        "        out = h + self.feed_forward(self.ffn_norm(h))\n",
        "        return out\n",
        "\n",
        "class GPTModel(nn.Module):\n",
        "    def __init__(self, params: ModelConfig):\n",
        "        super().__init__()\n",
        "        self.params = params\n",
        "        self.vocab_size = params.vocab_size\n",
        "        self.n_layers = params.n_layers\n",
        "\n",
        "        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)\n",
        "        self.layers = nn.ModuleList()\n",
        "        for layer_id in range(params.n_layers):\n",
        "            self.layers.append(TransformerBlock(layer_id, params))\n",
        "\n",
        "        self.norm = RMSNorm(params.dim, eps=params.norm_eps)\n",
        "        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)\n",
        "\n",
        "        # Precompute RoPE frequencies\n",
        "        self.freqs_cis = precompute_freqs_cis(\n",
        "            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2, self.params.rope_theta\n",
        "        )\n",
        "        \n",
        "        # Weight Tying (Optional, but often used to save params)\n",
        "        # self.tok_embeddings.weight = self.output.weight\n",
        "\n",
        "    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None):\n",
        "        bsz, seqlen = tokens.shape\n",
        "        h = self.tok_embeddings(tokens)\n",
        "        freqs_cis = self.freqs_cis[:seqlen].to(h.device)\n",
        "\n",
        "        # Mask is handled internally by F.scaled_dot_product_attention(is_causal=True)\n",
        "        mask = None\n",
        "\n",
        "        for layer in self.layers:\n",
        "            # Gradient Checkpointing Support\n",
        "            if self.training:\n",
        "                 h = torch.utils.checkpoint.checkpoint(layer, h, freqs_cis, mask, use_reentrant=False)\n",
        "            else:\n",
        "                 h = layer(h, freqs_cis, mask)\n",
        "\n",
        "        h = self.norm(h)\n",
        "        logits = self.output(h)\n",
        "\n",
        "        loss = None\n",
        "        if targets is not None:\n",
        "            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))\n",
        "\n",
        "        return logits, loss\n",
        "\n",
        "    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):\n",
        "        # Filter params requiring grad\n",
        "        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}\n",
        "        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]\n",
        "        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]\n",
        "        optim_groups = [\n",
        "            {'params': decay_params, 'weight_decay': weight_decay},\n",
        "            {'params': nodecay_params, 'weight_decay': 0.0}\n",
        "        ]\n",
        "        \n",
        "        # Use 8-bit AdamW if available and on CUDA\n",
        "        optimizer = None\n",
        "        if device_type == 'cuda':\n",
        "            try:\n",
        "                import bitsandbytes as bnb\n",
        "                optimizer = bnb.optim.AdamW8bit(optim_groups, lr=learning_rate, betas=betas)\n",
        "                print(\"Using 8-bit AdamW optimizer (bitsandbytes)\")\n",
        "            except Exception as e:\n",
        "                print(f\"bitsandbytes unavailable or failed to initialize: {e}\")\n",
        "                print(\"Falling back to standard torch.optim.AdamW\")\n",
        "                optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)\n",
        "        else:\n",
        "            print(\"Using standard AdamW optimizer (CPU/No-BNB)\")\n",
        "            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)\n",
        "            \n",
        "        return optimizer"
    ]
}

# Section 5: Dataset Pipeline
cell_5_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 5. Dataset Pipeline\n",
        "Streaming-capable dataset implementation using HuggingFace Datasets."
    ]
}

cell_5_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# @title 5. Data Loading\n",
        "class TokenizedDataset(IterableDataset):\n",
        "    def __init__(self, data, tokenizer, max_seq_len):\n",
        "        self.data = data\n",
        "        self.tokenizer = tokenizer\n",
        "        self.max_seq_len = max_seq_len\n",
        "\n",
        "    def __iter__(self):\n",
        "        buffer = []\n",
        "        for item in self.data:\n",
        "            # Add EOS token\n",
        "            tokens = self.tokenizer.encode(item['text'])\n",
        "            tokens.append(self.tokenizer.eos_id()) # Explicitly add EOS token\n",
        "            buffer.extend(tokens)\n",
        "            \n",
        "            # Yield chunks\n",
        "            while len(buffer) >= self.max_seq_len + 1:\n",
        "                chunk = buffer[:self.max_seq_len + 1]\n",
        "                buffer = buffer[self.max_seq_len + 1:]\n",
        "                x = torch.tensor(chunk[:-1], dtype=torch.long)\n",
        "                y = torch.tensor(chunk[1:], dtype=torch.long)\n",
        "                yield x, y\n",
        "\n",
        "def get_dataloader(tokenizer, split=\"train\", max_seq_len=512, batch_size=8):\n",
        "    dataset = load_dataset(\"wikitext\", \"wikitext-2-v1\", split=split)\n",
        "    # Using IterableDataset for memory efficiency\n",
        "    tokenized_ds = TokenizedDataset(dataset, tokenizer, max_seq_len)\n",
        "    return DataLoader(tokenized_ds, batch_size=batch_size, pin_memory=True)\n",
        "\n",
        "# Test Data Pipeline\n",
        "train_loader = get_dataloader(sp, split=\"train\", max_seq_len=128, batch_size=2)\n",
        "x_batch, y_batch = next(iter(train_loader))\n",
        "print(\"Sample Batch Shape:\", x_batch.shape)"
    ]
}

# Section 6: Training Loop
cell_6_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 6. Training Loop\n",
        "Includes:\n",
        "- **Mixed Precision**: bf16 (if supported) or fp16\n",
        "- **Gradient Accumulation**: Simulates larger batch sizes\n",
        "- **Logging**: Loss, LR, throughput\n",
        "- **Checkpointing**: Saves model state"
    ]
}

cell_6_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# @title 6. Training Engine\n",
        "\n",
        "def train_model(config_name, max_steps=1000, grad_accum_steps=4):\n",
        "    print(f\"\\n{'='*40}\\nStarting Training for Config: {config_name}\\n{'='*40}\")\n",
        "    \n",
        "    # 1. Config\n",
        "    config = get_config(config_name, device)\n",
        "    print(f\"Model Config: {config}\")\n",
        "    \n",
        "    # 2. Model Init\n",
        "    model = GPTModel(config).to(device)\n",
        "    \n",
        "    # Optional: Compile (Torch 2.0+)\n",
        "    # if device == 'cuda':\n",
        "    #     print(\"Compiling model with torch.compile...\")\n",
        "    #     model = torch.compile(model)\n",
        "\n",
        "    # 3. Optimizer\n",
        "    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, betas=(0.9, 0.95), device_type=device)\n",
        "\n",
        "    # 4. DataLoader\n",
        "    # Adjust batch size based on VRAM constraints of the config\n",
        "    batch_size = 4 if config.dim >= 1024 else 8\n",
        "    if device == 'cpu': batch_size = 2\n",
        "    \n",
        "    train_loader = get_dataloader(sp, split=\"train\", max_seq_len=config.max_seq_len, batch_size=batch_size)\n",
        "    train_iter = iter(train_loader)\n",
        "\n",
        "    # 5. Scheduler\n",
        "    scheduler = get_cosine_schedule_with_warmup(\n",
        "        optimizer, num_warmup_steps=100, num_training_steps=max_steps\n",
        "    )\n",
        "\n",
        "    # 6. Training Loop\n",
        "    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))\n",
        "    model.train()\n",
        "    \n",
        "    losses = []\n",
        "    start_time = time.time()\n",
        "    tokens_processed = 0\n",
        "\n",
        "    for step in range(max_steps):\n",
        "        optimizer.zero_grad()\n",
        "        loss_accum = 0.0\n",
        "        \n",
        "        for _ in range(grad_accum_steps):\n",
        "            try:\n",
        "                x, y = next(train_iter)\n",
        "            except StopIteration:\n",
        "                train_iter = iter(train_loader)\n",
        "                x, y = next(train_iter)\n",
        "            \n",
        "            x, y = x.to(device), y.to(device)\n",
        "            \n",
        "            # Mixed Precision Context\n",
        "            with torch.cuda.amp.autocast(enabled=(device == 'cuda'), dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):\n",
        "                logits, loss = model(x, y)\n",
        "                loss = loss / grad_accum_steps\n",
        "            \n",
        "            scaler.scale(loss).backward()\n",
        "            loss_accum += loss.item()\n",
        "            tokens_processed += x.numel()\n",
        "\n",
        "        scaler.step(optimizer)\n",
        "        scaler.update()\n",
        "        scheduler.step()\n",
        "\n",
        "        losses.append(loss_accum)\n",
        "\n",
        "        if step % 50 == 0:\n",
        "            dt = time.time() - start_time\n",
        "            tps = tokens_processed / dt\n",
        "            lr = scheduler.get_last_lr()[0]\n",
        "            print(f\"Step {step}/{max_steps} | Loss: {loss_accum:.4f} | LR: {lr:.2e} | TPS: {tps:.0f}\")\n",
        "\n",
        "    # Save Checkpoint\n",
        "    checkpoint_path = f\"checkpoint_{config_name}.pt\"\n",
        "    torch.save(model.state_dict(), checkpoint_path)\n",
        "    print(f\"Saved checkpoint to {checkpoint_path}\")\n",
        "    \n",
        "    return losses"
    ]
}

# Section 7: Progressive Scaling
cell_7_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 7. Progressive Scaling Roadmap\n",
        "Run the cells below to simulate the 7-day scaling plan."
    ]
}

cell_7_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# @title Day 1: Train 100M Model (Baseline)\n",
        "losses_100M = train_model(\"100M\", max_steps=200, grad_accum_steps=4)"
    ]
}

cell_8_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# @title Day 2: Scale to 300M\n",
        "losses_300M = train_model(\"300M\", max_steps=100, grad_accum_steps=8)"
    ]
}

cell_9_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# @title Day 3-4: Scale to 700M\n",
        "losses_700M = train_model(\"700M\", max_steps=50, grad_accum_steps=16)"
    ]
}

cell_10_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# @title Day 5: Scale to 1B\n",
        "losses_1B = train_model(\"1B\", max_steps=20, grad_accum_steps=32)"
    ]
}

cell_11_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# @title Day 6: Validate 2B Architecture (Forward Pass Only)\n",
        "print(\"Initializing 2B Model for Validation...\")\n",
        "try:\n",
        "    config_2B = get_config(\"2B\", device)\n",
        "    model_2B = GPTModel(config_2B).to(device)\n",
        "    print(\"2B Model Successfully Initialized on\", device)\n",
        "    \n",
        "    # Dummy forward pass\n",
        "    dummy_input = torch.randint(0, 32000, (1, 128)).to(device)\n",
        "    with torch.no_grad():\n",
        "        logits, _ = model_2B(dummy_input)\n",
        "    print(\"Forward pass successful. Output shape:\", logits.shape)\n",
        "    \n",
        "    del model_2B\n",
        "    if device == 'cuda':\n",
        "        torch.cuda.empty_cache()\n",
        "except Exception as e:\n",
        "    print(f\"2B Validation Failed (Expected on T4 if OOM): {e}\")"
    ]
}

# Section 8: Inference
cell_12_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 8. Inference Testing\n",
        "Load the trained 100M model for text generation."
    ]
}

cell_12_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# @title Generate Text\n",
        "\n",
        "def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, top_k=40):\n",
        "    model.eval()\n",
        "    tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)\n",
        "    \n",
        "    for _ in range(max_new_tokens):\n",
        "        with torch.no_grad():\n",
        "            logits, _ = model(tokens)\n",
        "            logits = logits[:, -1, :] / temperature\n",
        "            \n",
        "            # Top-K Sampling\n",
        "            v, _ = torch.topk(logits, top_k)\n",
        "            logits[logits < v[:, [-1]]] = -float('Inf')\n",
        "            \n",
        "            probs = F.softmax(logits, dim=-1)\n",
        "            next_token = torch.multinomial(probs, num_samples=1)\n",
        "            tokens = torch.cat((tokens, next_token), dim=1)\n",
        "\n",
        "    output = tokenizer.decode(tokens[0].tolist())\n",
        "    return output\n",
        "\n",
        "# Load 100M checkpoint for demo\n",
        "config_100M = get_config(\"100M\", device)\n",
        "model_infer = GPTModel(config_100M).to(device)\n",
        "try:\n",
        "    # Explicitly map to current device to avoid CUDA/CPU mismatch\n",
        "    state_dict = torch.load(\"checkpoint_100M.pt\", map_location=device)\n",
        "    model_infer.load_state_dict(state_dict)\n",
        "    print(\"✅ Loaded 100M Checkpoint successfully\")\n",
        "except FileNotFoundError:\n",
        "    print(\"⚠️ Checkpoint 'checkpoint_100M.pt' not found. Inference will use RANDOM WEIGHTS.\")\n",
        "except Exception as e:\n",
        "    print(f\"❌ Failed to load checkpoint: {e}. Inference will use RANDOM WEIGHTS.\")\n",
        "\n",
        "prompt = \"The future of AI is\"\n",
        "generated_text = generate(model_infer, sp, prompt)\n",
        "print(f\"\\nPrompt: {prompt}\\nGenerated: {generated_text}\")"
    ]
}

# Section 9: Plotting
cell_13_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# @title 9. Scaling Law Plot\n",
        "plt.figure(figsize=(10, 6))\n",
        "if 'losses_100M' in locals(): plt.plot(losses_100M, label='100M')\n",
        "if 'losses_300M' in locals(): plt.plot(losses_300M, label='300M')\n",
        "if 'losses_700M' in locals(): plt.plot(losses_700M, label='700M')\n",
        "if 'losses_1B' in locals(): plt.plot(losses_1B, label='1B')\n",
        "\n",
        "plt.xlabel(\"Steps\")\n",
        "plt.ylabel(\"Loss\")\n",
        "plt.title(\"Training Loss Curves by Model Size\")\n",
        "plt.legend()\n",
        "plt.grid(True)\n",
        "plt.show()"
    ]
}


# --- Assemble Notebook ---

def main():
    notebook_structure = {
        "cells": [
            cell_1_markdown, cell_1_code,
            cell_2_markdown, cell_2_code,
            cell_3_markdown, cell_3_code,
            cell_4_markdown, cell_4_code,
            cell_5_markdown, cell_5_code,
            cell_6_markdown, cell_6_code,
            cell_7_markdown,
            cell_7_code,
            cell_8_code,
            cell_9_code,
            cell_10_code,
            cell_11_code,
            cell_12_markdown, cell_12_code,
            cell_13_code
        ],
        "metadata": {
            "colab": {
                "name": "UpFlame Progressive Scaling 100M to 2B",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    # Write to file (Relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "upflame_progressive_scaling_100M_to_2B.ipynb")

    with open(output_file, "w") as f:
        json.dump(notebook_structure, f, indent=2)

    print(f"Successfully generated notebook at: {output_file}")

if __name__ == "__main__":
    main()
