import torch
import os
import sys
from datasets import load_dataset
try:
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
except ImportError:
    print("TRL library not found. Install with `pip install trl`.")
    sys.exit(0)

from upflame_ago.model.config import UpFlameAGOConfig
from upflame_ago.model.architecture import UpFlameAGOForCausalLM
from transformers import AutoTokenizer, LlamaTokenizer

def main():
    tokenizer_path = "upflame_ago_tokenizer.model"
    vocab_size = 65536

    if os.path.exists(tokenizer_path):
        print(f"Loading custom tokenizer from {tokenizer_path}...")
        try:
            tokenizer = LlamaTokenizer(vocab_file=tokenizer_path)
            tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            print(f"Failed to load custom tokenizer: {e}. Falling back to GPT2.")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            vocab_size = tokenizer.vocab_size
    else:
        print("Custom tokenizer not found. Falling back to GPT2.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size

    training_args_dict = {
        "learning_rate": 1.41e-5,
        "per_device_train_batch_size": 1,
        "logging_steps": 1,
    }
    if not torch.cuda.is_available():
        training_args_dict["use_cpu"] = True

    config = PPOConfig(
        mini_batch_size=1,
        batch_size=1,
        **training_args_dict
    )

    print(f"Initializing RL policy model with vocab_size={vocab_size}...")

    base_model = UpFlameAGOForCausalLM(UpFlameAGOConfig(vocab_size=vocab_size, hidden_size=256, num_hidden_layers=2))
    ref_base_model = UpFlameAGOForCausalLM(UpFlameAGOConfig(vocab_size=vocab_size, hidden_size=256, num_hidden_layers=2))

    model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ref_base_model)

    try:
        dataset = load_dataset("imdb", split="train[:10]")
    except:
        from datasets import Dataset
        dataset = Dataset.from_dict({"text": ["This movie is bad."] * 10})

    dataset = dataset.rename_column("text", "query")
    def truncate(x):
        return {"query": x["query"][:50]}
    dataset = dataset.map(truncate, batched=False)

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    # TRL Experimental PPO uses `args` or `config`.
    # It might use `processing_class` instead of `tokenizer` in newer versions.
    # Check TRL experimental PPO constructor signature.
    # It usually is (config, model, ref_model, tokenizer, dataset, data_collator)
    # BUT `PPOTrainer.__init__` might be strict.
    # If using experimental, the signature is often: (args, model, ref_model, processing_class, ...)

    try:
         ppo_trainer = PPOTrainer(
            args=config,
            model=model,
            ref_model=ref_model,
            processing_class=tokenizer, # New name for tokenizer
            train_dataset=dataset,
            data_collator=collator
        )
    except TypeError:
        # Fallback to older
         ppo_trainer = PPOTrainer(
            config=config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=dataset,
            data_collator=collator
        )

    print("Starting PPO loop...")
    for epoch, batch in enumerate(ppo_trainer.dataloader):
        query_tensors = [tokenizer(q, return_tensors="pt")["input_ids"][0] for q in batch["query"]]

        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            max_new_tokens=20
        )
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        rewards = [torch.tensor(1.0) if "good" in r else torch.tensor(-1.0) for r in batch["response"]]

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        print(f"Step {epoch} complete. Mean Reward: {sum(rewards)/len(rewards)}")
        if epoch >= 2: break

if __name__ == "__main__":
    main()
