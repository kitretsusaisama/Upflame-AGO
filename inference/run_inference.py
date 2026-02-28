import argparse
import os
import sys
import torch
import yaml

# Ensure we can import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import math

from model.unified_config import UpFlameAGOUnifiedConfig
from model.unified_transformer import UnifiedTransformer
from tokenizer.tokenizer import UpFlameAGOTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Inference on a trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory (e.g., checkpoints/100M)")
    parser.add_argument("--prompt", type=str, default="The future of open source AI is", help="Prompt to start generation")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--evaluate", action="store_true", help="Run precise MNC-grade perplexity evaluation on the prompt instead of generation")
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
    tokenizer = UpFlameAGOTokenizer(model_path=tokenizer_path)
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

    # 4. Execute (Generation or Evaluation)
    logger.info(f"Prompt: '{args.prompt}'")
    
    enc = tokenizer.encode(args.prompt, add_special_tokens=False)
    input_tokens = enc['input_ids'] if isinstance(enc, dict) else enc
    input_ids = torch.tensor([input_tokens], dtype=torch.long).to(device)

    if args.evaluate:
        logger.info("Running MNC-Grade precise perplexity evaluation...")
        if input_ids.size(1) < 2:
            logger.error("❌ Evaluation requires a prompt with at least 2 tokens to compute valid loss.")
            return

        with torch.no_grad():
            outputs = model(input_ids=input_ids, return_dict=True)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            perplexity = math.exp(loss.item())
            
        logger.info("-" * 50)
        logger.info(f"📊 Evaluation Results:")
        logger.info(f"Cross-Entropy Loss : {loss.item():.4f}")
        logger.info(f"Exact Perplexity   : {perplexity:.4f}")
        logger.info("-" * 50)

    else:
        logger.info("Generating...")
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
