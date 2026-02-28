import argparse
import os
import sys
import torch
import yaml

# Ensure we can import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from transformers import AutoTokenizer

from model.unified_config import UpFlameAGOUnifiedConfig
from model.unified_transformer import UnifiedTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Inference on a trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory (e.g., checkpoints/100M)")
    parser.add_argument("--prompt", type=str, default="The future of open source AI is", help="Prompt to start generation")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    args = parser.parse_args()

    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        logger.info(f"✅ TPU Detected: {device}")
    except ImportError:
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.warning("⚠️ No GPU detected. CPU fallback mode will be engaged.")

    # Determine scale from checkpoint path
    scale = os.path.basename(os.path.normpath(args.checkpoint))
    
    # 1. Load Config
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "scaling.yaml")
    try:
        with open(config_path, "r") as f:
            scaling_configs = yaml.safe_load(f)["scales"]
    except FileNotFoundError:
        logger.error(f"Error: Could not find {config_path}.")
        return

    if scale not in scaling_configs:
        logger.warning(f"Warning: Scale '{scale}' not found in scaling.yaml. Assuming 100M defaults for architecture.")
        preset = scaling_configs["100M"]
    else:
        preset = scaling_configs[scale]

    logger.info(f"Initializing architecture for scale: {scale} in Baseline Mode")
    UpFlameAGOUnifiedConfig.USE_ADVANCED = False

    # 2. Setup Tokenizer
    tokenizer_path = os.path.join(os.path.dirname(__file__), "..", "tokenizer", "upflame_ago_tokenizer.model")
    if os.path.exists(tokenizer_path):
        import sentencepiece as spm
        class SPMTokenizerWrap:
            def __init__(self, spm_model_path):
                self.sp = spm.SentencePieceProcessor()
                self.sp.load(spm_model_path)
                self.vocab_size = self.sp.vocab_size()
            def encode(self, text, add_special_tokens=False):
                return self.sp.encode(text)
            def decode(self, token_ids):
                return self.sp.decode(token_ids)
        logger.info("Loading native SentencePiece tokenizer...")
        tokenizer = SPMTokenizerWrap(tokenizer_path)
        vocab_size = tokenizer.vocab_size
    else:
        logger.warning(f"Native tokenizer not found at {tokenizer_path}. Falling back to GPT-2 tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        vocab_size = tokenizer.vocab_size

    model_config = UpFlameAGOUnifiedConfig(
        vocab_size=vocab_size,
        hidden_size=preset["hidden_size"],
        num_hidden_layers=preset["layers"],
        num_attention_heads=preset["heads"],
        num_key_value_heads=preset["heads"] // 2 if preset["heads"] % 2 == 0 else preset["heads"],
        max_position_embeddings=preset["context"]
    )

    model = UnifiedTransformer(model_config).to(device)

    # 3. Load Checkpoint
    ckpt_file = os.path.join(args.checkpoint, "model.pt")
    try:
        logger.info(f"Loading checkpoint from {ckpt_file}...")
        state_dict = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(state_dict)
        logger.info("✅ Checkpoint loaded successfully.")
    except FileNotFoundError:
        logger.error(f"❌ Error: Checkpoint file not found at {ckpt_file}")
        return
    except Exception as e:
        logger.error(f"❌ Error loading checkpoint: {e}")
        return

    model.eval()

    # 4. Generate
    logger.info(f"Prompt: '{args.prompt}'")
    logger.info("Generating...")
    
    enc = tokenizer.encode(args.prompt, add_special_tokens=False)
    input_tokens = enc['input_ids'] if isinstance(enc, dict) else enc
    input_ids = torch.tensor([input_tokens], dtype=torch.long).to(device)

    for _ in range(args.max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :] / args.temperature
            
            # Simple Top-K
            top_k = 40
            v, _ = torch.topk(next_token_logits, top_k)
            next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat((input_ids, next_token), dim=1)

    try:
        generated_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
    except TypeError:
        # Fallback if sentencepiece decode doesn't accept skip_special_tokens param directly
        generated_text = tokenizer.decode(input_ids[0].tolist())

    logger.info(f"\nFinal Output: \n{generated_text}")

if __name__ == "__main__":
    main()
