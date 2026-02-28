import argparse
import yaml
import os
import sys
import torch
import json
import time

# Ensure we can import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.unified_config import UpFlameAGOUnifiedConfig
from model.unified_transformer import UnifiedTransformer
from datasets import load_dataset, interleave_datasets
from torch.utils.data import DataLoader, IterableDataset

import logging
from tokenizer.tokenizer import UpFlameAGOTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class TokenizedDataset(IterableDataset):
    def __init__(self, data, tokenizer, max_seq_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __iter__(self):
        buffer = []
        for item in self.data:
            # Handle different dataset structures (wikitext uses 'text', some code datasets use 'content' or 'instruction'/'output')
            text = item.get('text', '')
            if not text:
                text = item.get('content', '')
            if not text:
                instruction = item.get('instruction', '')
                inp = item.get('input', '')
                output = item.get('output', '')

                parts = [instruction, inp, output]
                text = "\n".join([str(p).strip() for p in parts if p])

            if not text.strip():
                continue

            # If using huggingface tokenizer, encode returns a list directly or a dict
            enc = self.tokenizer.encode(text, add_special_tokens=False)
            if isinstance(enc, dict):
                tokens = enc['input_ids']
            else:
                tokens = enc

            # Use appropriate eos token id
            eos = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 2
            tokens.append(eos)
            buffer.extend(tokens)

            while len(buffer) >= self.max_seq_len + 1:
                chunk = buffer[:self.max_seq_len + 1]
                buffer = buffer[self.max_seq_len + 1:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y

def main():
    parser = argparse.ArgumentParser(description="Progressive Transformer Training for Colab")
    parser.add_argument("--scale", type=str, required=True, choices=["100M", "300M", "700M", "1B", "2B"], help="Model scale preset")
    parser.add_argument("--cpu_mode", action="store_true", help="Force CPU training mode with reduced context/batch size")
    parser.add_argument("--validate_only", action="store_true", help="Only build model and run one forward pass (for 2B testing)")
    parser.add_argument("--max_steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--use_wandb", action="store_true", help="Enable tracking with Weights & Biases")
    
    args = parser.parse_args()

    # Device detection logic
    if args.cpu_mode:
        device = "cpu"
        logger.warning("⚠️ CPU mode requested. CPU fallback mode will be engaged.")
    else:
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

    # 1. Load Scaling Config
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "scaling.yaml")
    try:
        with open(config_path, "r") as f:
            scaling_configs = yaml.safe_load(f)["scales"]
    except FileNotFoundError:
        print(f"Error: Could not find {config_path}. Did you create it?")
        return

    if args.scale not in scaling_configs:
        print(f"Error: Scale {args.scale} not found in configs/scaling.yaml")
        return

    preset = scaling_configs[args.scale]

    # Adjust for CPU mode
    context_len = 512 if args.cpu_mode else preset["context"]
    batch_size = 2 if args.cpu_mode else (4 if preset["hidden_size"] >= 1024 else 8)
    grad_accum_steps = 2 if args.cpu_mode else 4

    logger.info(f"--- Configuration for {args.scale} ---")
    logger.info(f"Layers: {preset['layers']}, Hidden Size: {preset['hidden_size']}, Heads: {preset['heads']}, Context: {context_len}")

    # 2. Setup Tokenizer FIRST to get correct vocab_size
    # We pass None here to allow UpFlameAGOTokenizer to safely resolve the path using CWD and __file__ bounds.
    tokenizer = UpFlameAGOTokenizer(model_path=None)
    vocab_size = tokenizer.vocab_size

    
    # 3. Initialize Model Config (Disabling Advanced Features for Week 1 Colab)
    model_config = UpFlameAGOUnifiedConfig(
        vocab_size=vocab_size,
        hidden_size=preset["hidden_size"],
        num_hidden_layers=preset["layers"],
        num_attention_heads=preset["heads"],
        num_key_value_heads=preset["heads"] // 2 if preset["heads"] % 2 == 0 else preset["heads"], # GQA or standard
        max_position_embeddings=context_len,
        use_moe=False, # DISABLED FOR COLAB BASELINE
        use_infini_attention=False, # DISABLED FOR COLAB BASELINE
        use_vector_memory=False, # DISABLED FOR COLAB BASELINE
        use_world_state=False # DISABLED FOR COLAB BASELINE
    )

    logger.info("Initializing UnifiedTransformer in Baseline Mode...")
    model = UnifiedTransformer(model_config)
    logger.info("Transferring model to device...")
    model = model.to(device)
    logger.info("Model transferred to device successfully.")

    if args.validate_only:
        logger.info("Validation Mode: Running single forward pass...")
        dummy_input = torch.randint(0, model_config.vocab_size, (1, 128)).to(device)
        try:
            with torch.no_grad():
                outputs = model(input_ids=dummy_input, return_dict=True)
            logger.info(f"✅ Forward pass successful. Output shape: {outputs.logits.shape}")
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
        
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        return

    # 4. Setup Optimizer with Fallback
    try:
        if device == "cuda":
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=3e-4)
            print("Using 8-bit AdamW optimizer")
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            print("Using standard torch.optim.AdamW")
    except Exception as e:
        print(f"Failed to initialize bitsandbytes optimizer: {e}. Falling back to standard AdamW.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # 5. Setup Data Pipeline

    try:
        logger.info("Initiating Dataset Load (This may take a few minutes in Colab to download from HuggingFace)...")
        # For interleave_datasets, columns must match.
        # So we map the code dataset to have a single 'text' column just like wikitext.
        def format_code(example):
            instruction = example.get('instruction', '')
            inp = example.get('input', '')
            output = example.get('output', '')
            parts = [instruction, inp, output]
            return {"text": "\n".join([str(p).strip() for p in parts if p])}

        lang_dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
        code_dataset = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train")
        code_dataset = code_dataset.map(format_code, remove_columns=code_dataset.column_names)

        # Interleave datasets to mix language and code
        dataset = interleave_datasets([lang_dataset, code_dataset])
        logger.info("✅ Datasets loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}. Attempting fallback to synthetic data.")
        # Fallback dataset for absolute resilience (MNC grade)
        dataset = [{"text": "Synthetic fallback data point to ensure training does not crash."} for _ in range(1000)]

    tokenized_ds = TokenizedDataset(dataset, tokenizer, max_seq_len=context_len)
    train_loader = DataLoader(tokenized_ds, batch_size=batch_size)
    train_iter = iter(train_loader)

    # 5. Training Loop
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project="upflame-ago-training",
                name=f"baseline-{args.scale}-{device}",
                config={
                    "scale": args.scale,
                    "device": device,
                    "max_steps": args.max_steps,
                    "batch_size": batch_size,
                    "context_len": context_len,
                    "grad_accum_steps": grad_accum_steps
                }
            )
            logger.info("✅ Weights & Biases initialized.")
        except ImportError:
            logger.warning("⚠️ wandb is not installed. Run `pip install wandb` to use tracking. Proceeding without wandb.")
            args.use_wandb = False
        except Exception as e:
            logger.error(f"❌ Failed to initialize Weights & Biases: {e}. Proceeding without it.")
            args.use_wandb = False

    logger.info(f"Starting training for {args.max_steps} steps...")
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    model.train()

    logs = []
    start_time = time.time()

    # Ensure checkpoint and log directories exist
    os.makedirs(f"checkpoints/{args.scale}", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    try:
        for step in range(args.max_steps):
            optimizer.zero_grad()
            loss_accum = 0.0
            
            for _ in range(grad_accum_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)

                x, y = x.to(device), y.to(device)
                
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    # UnifiedTransformer returns a CausalLMOutputWithPast or tuple
                    outputs = model(input_ids=x, return_dict=True)
                    logits = outputs.logits

                    # Manual cross entropy loss calculation
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / grad_accum_steps

                scaler.scale(loss).backward()
                loss_accum += loss.item()
                
            scaler.step(optimizer)
            scaler.update()
            
            if str(device).startswith("xla"):
                import torch_xla.core.xla_model as xm
                xm.mark_step()

            logs.append({"step": step, "loss": loss_accum})
            if args.use_wandb:
                wandb.log({"loss": loss_accum, "step": step})

            if step % 10 == 0:
                logger.info(f"Step {step}/{args.max_steps} | Loss: {loss_accum:.4f}")

        # Save Checkpoint and Logs
        ckpt_path = f"checkpoints/{args.scale}/model.pt"
        torch.save(model.state_dict(), ckpt_path)

        # Prominent completion notification with terminal beep
        print("\a") # Terminal beep
        print("=" * 50)
        logger.info(f"🎉 TRAINING COMPLETE! 🎉")
        logger.info(f"✅ Checkpoint successfully saved to: {ckpt_path}")
        print("=" * 50)

        with open(f"logs/scaling_{args.scale}.json", "w") as f:
            json.dump(logs, f)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving checkpoint and exiting gracefully...")
        ckpt_path = f"checkpoints/{args.scale}/model_interrupted.pt"
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f"Interrupt checkpoint saved to {ckpt_path}")
    except Exception as e:
        logger.error(f"Training failed due to unexpected error: {e}", exc_info=True)
    finally:
        if args.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    main()
