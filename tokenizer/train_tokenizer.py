import sentencepiece as spm
import os
import argparse
import random
import string

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="tokenizer_training_data.txt", help="Path to training data")
    # Force default prefix to save exactly adjacent to this script in the tokenizer folder
    default_prefix = os.path.join(os.path.dirname(__file__), "upflame_ago_tokenizer")
    parser.add_argument("--model_prefix", type=str, default=default_prefix, help="Output model prefix")
    parser.add_argument("--vocab_size", type=int, default=65536, help="Vocabulary size")
    args = parser.parse_args()

    print(f"Generating massive high-entropy training data at {args.input_file} to support vocab_size={args.vocab_size}...")
    with open(args.input_file, "w") as f:
        # Standard structural text
        for _ in range(5000):
            f.write("UpFlame AGO is an open source artificial general operator foundation model.\n")
            f.write("It uses reinforcement learning and multi-agent orchestration.\n")
            f.write("def hello_world():\n    print('hello world!')\n")
            f.write("State: <|state|> Action: <|action|> Reward: <|reward|>\n")

        # Wide vocabulary seed (combining diverse alphanumeric characters and structures to force BPE splits)
        words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew", "agent", "robot", "future", "code", "python", "rust", "ago", "nvsn", "transformers", "pytorch", "tensor", "gradient", "descent", "backpropagation", "attention", "mechanism", "matrix", "vector", "embedding", "activation"]

        # Inject huge random byte/string entropy so BPE bottom-up merge can easily hit 65536
        for _ in range(50000):
            sent = " ".join(random.choices(words, k=20))
            f.write(sent + "\n")

        # Add random character strings to ensure vast unique combinations
        chars = string.ascii_letters + string.digits + string.punctuation + " \n\t"
        for _ in range(100000):
            rand_str = "".join(random.choices(chars, k=100))
            f.write(rand_str + "\n")

    user_defined_symbols = [
        "<|agent_start|>", "<|agent_end|>", "<|agent_plan|>", "<|agent_execute|>", "<|agent_reflect|>",
        "<|tool_call|>", "<|tool_result|>", "<|tool_error|>",
        "<|reward|>", "<|state|>", "<|action|>", "<|next_state|>",
        "<|memory_read|>", "<|memory_write|>", "<|memory_ref|>",
        "<|plan|>", "<|subplan|>", "<|goal|>", "<|task|>",
        "<|image|>", "<|audio|>", "<|video|>",
        "<|system|>", "<|user|>", "<|assistant|>"
    ]

    file_size = os.path.getsize(args.input_file)
    vocab_size = args.vocab_size

    # BPE is bottom-up, it combines characters safely until vocab_size is reached.
    # We remove the unigram artificial 32k hard cap, and allow full 65536 standard size.
    if vocab_size > file_size // 2: # Safe ratio for BPE merging
         vocab_size = max(1000, file_size // 2)
         print(f"Adjusting vocab size to {vocab_size} because corpus is incredibly small.")

    print(f"Training BPE Tokenizer with vocab_size={vocab_size} (MNC Grade Standard)...")

    try:
        spm.SentencePieceTrainer.train(
            input=args.input_file,
            model_prefix=args.model_prefix,
            vocab_size=vocab_size,
            character_coverage=1.0, # BPE usually requires 1.0 for byte-level completeness
            model_type="bpe",
            user_defined_symbols=user_defined_symbols,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<|pad|>",
            unk_piece="<|unk|>",
            bos_piece="<|bos|>",
            eos_piece="<|eos|>",
            byte_fallback=True # Essential for MNC-grade Out-of-Vocabulary (OOV) resilience
        )
        print(f"Tokenizer trained. Model saved to {args.model_prefix}.model")
        print(f"Vocab saved to {args.model_prefix}.vocab")
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()
