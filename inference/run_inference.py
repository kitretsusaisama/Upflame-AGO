import argparse
import os
import sys
import torch
import yaml

# Ensure we can import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.unified_config import UpFlameAGOUnifiedConfig
from model.unified_transformer import UnifiedTransformer

# Simple dummy tokenizer matchin train_small.py
class DummyTokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.eos_id = 2
        self.pad_id = 0
        # Invert mapping for decoding demo
        self.vocab = {i: f"token_{i}" for i in range(vocab_size)}
        # Basic common words hack for demo readability
        for word in ["The", "future", "of", "open", "source", "AI", "is", "bright"]:
            self.vocab[hash(word) % self.vocab_size] = word

    def encode(self, text):
        return [hash(w) % self.vocab_size for w in text.split()]
        
    def decode(self, token_ids):
        return " ".join([self.vocab.get(tid, f"<unk_{tid}>") for tid in token_ids])

def main():
    parser = argparse.ArgumentParser(description="Run Inference on a trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory (e.g., checkpoints/100M)")
    parser.add_argument("--prompt", type=str, default="The future of open source AI is", help="Prompt to start generation")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Determine scale from checkpoint path
    scale = os.path.basename(os.path.normpath(args.checkpoint))
    
    # 1. Load Config
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "scaling.yaml")
    try:
        with open(config_path, "r") as f:
            scaling_configs = yaml.safe_load(f)["scales"]
    except FileNotFoundError:
        print(f"Error: Could not find {config_path}.")
        return

    if scale not in scaling_configs:
        print(f"Warning: Scale '{scale}' not found in scaling.yaml. Assuming 100M defaults for architecture.")
        preset = scaling_configs["100M"]
    else:
        preset = scaling_configs[scale]

    print(f"Initializing architecture for scale: {scale} in Baseline Mode")
    UpFlameAGOUnifiedConfig.USE_ADVANCED = False
    model_config = UpFlameAGOUnifiedConfig(
        vocab_size=32000,
        hidden_size=preset["hidden_size"],
        num_hidden_layers=preset["layers"],
        num_attention_heads=preset["heads"],
        num_key_value_heads=preset["heads"] // 2 if preset["heads"] % 2 == 0 else preset["heads"],
        max_position_embeddings=preset["context"]
    )

    model = UnifiedTransformer(model_config).to(device)

    # 2. Load Checkpoint
    ckpt_file = os.path.join(args.checkpoint, "model.pt")
    try:
        print(f"Loading checkpoint from {ckpt_file}...")
        state_dict = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(state_dict)
        print("✅ Checkpoint loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: Checkpoint file not found at {ckpt_file}")
        return
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return

    model.eval()
    tokenizer = DummyTokenizer()

    # 3. Generate
    print(f"\nPrompt: '{args.prompt}'")
    print("Generating...")
    
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long).to(device)

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

    generated_text = tokenizer.decode(input_ids[0].tolist())
    print(f"\nFinal Output: \n{generated_text}")

if __name__ == "__main__":
    main()
