import sentencepiece as spm
import os
import argparse
import random
import string

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="tokenizer_training_data.txt", help="Path to training data")
    parser.add_argument("--model_prefix", type=str, default="upflame_ago_tokenizer", help="Output model prefix")
    parser.add_argument("--vocab_size", type=int, default=65536, help="Vocabulary size")
    args = parser.parse_args()

    print(f"Generating rich dummy training data at {args.input_file}...")
    with open(args.input_file, "w") as f:
        for _ in range(2000):
            f.write("UpFlame AGO is an open source artificial general operator foundation model.\n")
            f.write("It uses reinforcement learning and multi-agent orchestration.\n")
            f.write("def hello_world():\n  print('hello')\n")
            f.write("State: <|state|> Action: <|action|> Reward: <|reward|>\n")

        words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew", "agent", "robot", "future", "code", "python", "rust", "ago", "nvsn"]
        for _ in range(5000):
            sent = " ".join(random.choices(words, k=10))
            f.write(sent + "\n")
            rand_str = "".join(random.choices(string.ascii_letters, k=50))
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
    if vocab_size > file_size // 10:
         vocab_size = max(1000, file_size // 10)
         print(f"Adjusting vocab size to {vocab_size} based on data size.")

    print(f"Training tokenizer with vocab_size={vocab_size}...")

    try:
        spm.SentencePieceTrainer.train(
            input=args.input_file,
            model_prefix=args.model_prefix,
            vocab_size=vocab_size,
            character_coverage=0.9995,
            model_type="unigram",
            user_defined_symbols=user_defined_symbols,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<|pad|>",
            unk_piece="<|unk|>",
            bos_piece="<|bos|>",
            eos_piece="<|eos|>"
        )
        print(f"Tokenizer trained. Model saved to {args.model_prefix}.model")
        print(f"Vocab saved to {args.model_prefix}.vocab")
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()
