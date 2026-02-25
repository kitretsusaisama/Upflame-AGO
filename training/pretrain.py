import sys
import os
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Ensure we can import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from upflame_ago.model.config import UpFlameAGOConfig
from upflame_ago.model.architecture import UpFlameAGOForCausalLM
from transformers import PreTrainedTokenizerFast

def main():
    # 1. Initialize Config
    config = UpFlameAGOConfig(
        vocab_size=65536,
        hidden_size=512, # Small for demo
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=1024
    )

    # 2. Initialize Model
    print("Initializing model...")
    model = UpFlameAGOForCausalLM(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # 3. Load Tokenizer (dummy load, normally load from file)
    # Since we trained a tokenizer in step 3, we can load it if it exists.
    tokenizer_path = "upflame_ago_tokenizer.model"
    if os.path.exists(tokenizer_path):
        from transformers import LlamaTokenizer
        # SentencePiece model can be loaded via LlamaTokenizer (hacky but works)
        # Or PreTrainedTokenizerFast
        try:
            tokenizer = LlamaTokenizer(vocab_file=tokenizer_path)
            tokenizer.pad_token = tokenizer.eos_token
        except:
             print("Could not load tokenizer properly. Using dummy.")
             tokenizer = None
    else:
        print("Tokenizer not found. Skipping tokenizer load.")
        tokenizer = None

    # 4. Load Dataset (dummy)
    # In production, use "c4" or "wikitext"
    print("Loading dataset...")
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:100]") # Tiny slice
    except:
        # Fallback if internet fails or dataset issues
        from datasets import Dataset
        dataset = Dataset.from_dict({"text": ["UpFlame AGO is great."] * 100})

    def tokenize_function(examples):
        if tokenizer:
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        # Manually create labels for Causal LM training (labels = input_ids)
        # Since trainer does it automatically if data_collator is DataCollatorForLanguageModeling(mlm=False)
        # But if tokenizer is None, we need to handle it.
        return {"input_ids": [[1, 2, 3]] * len(examples["text"]), "labels": [[1, 2, 3]] * len(examples["text"])}

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=100,
        logging_steps=10,
        report_to="none", # Disable wandb for demo
        use_cpu=not torch.cuda.is_available()
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False) if tokenizer else None,
    )

    # 7. Train
    print("Starting training...")
    trainer.train()
    print("Training complete.")
    trainer.save_model("./upflame_ago_1b")

if __name__ == "__main__":
    main()
