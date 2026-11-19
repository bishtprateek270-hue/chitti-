"""
Chitti Fine-Tuning Script using QLoRA / LoRA (PEFT + TRL).
Optimized for local training on NVIDIA RTX GPUs (e.g. RTX 5050 8GB VRAM) or Google Colab.

Features:
- Real-time training monitoring with tqdm loss & learning rate display.
- 4-bit NormalFloat (NF4) quantization to fit within 6-8 GB VRAM.
- LoRA rank-16 target module adaptation across attention & MLP projections.
- Post-training generation test to verify personality & memory grounding.
"""

import os
import sys
import argparse
import time
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Chitti's LLM brain using LoRA")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-Coder-3B-Instruct",
        help="Base Hugging Face model ID (e.g., 'Qwen/Qwen2.5-Coder-3B-Instruct' or 'Qwen/Qwen2.5-Coder-7B-Instruct')"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/training/chitti_train.jsonl",
        help="Path to training jsonl file"
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="data/training/chitti_val.jsonl",
        help="Path to validation jsonl file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/chitti_lora_adapter",
        help="Output directory to save trained LoRA adapter"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device train batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = Path(__file__).resolve().parent.parent
    os.chdir(root_dir)

    print("\n" + "=" * 65)
    print("      C H I T T I   L L M   F I N E - T U N I N G   P I P E L I N E")
    print("=" * 65)
    print(f"[*] Base Model        : {args.model_id}")
    print(f"[*] Training Data     : {args.train_file}")
    print(f"[*] Output Directory  : {args.output_dir}")
    print(f"[*] Epochs            : {args.epochs}")
    print(f"[*] Batch Size        : {args.batch_size} (effective: {args.batch_size * args.grad_accum})")
    print(f"[*] Learning Rate     : {args.lr}")
    print(f"[*] 4-bit QLoRA       : {'Disabled' if args.no_4bit else 'Enabled (NF4)'}")
    print("=" * 65 + "\n")

    # Dependency check
    try:
        import torch
        from datasets import load_dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        print(f"[!] Missing required library: {e}")
        print("\nPlease install the required fine-tuning dependencies using:")
        print("    pip install torch transformers datasets peft trl bitsandbytes accelerate")
        sys.exit(1)

    # 1. Device check & GPU diagnostics
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[+] CUDA Available! Using GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    else:
        print("[!] Warning: CUDA is not available. Training on CPU will be extremely slow.")

    # 2. Tokenizer setup
    print("\n[1/5] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Quantization config (QLoRA)
    bnb_config = None
    has_bnb = False
    try:
        import bitsandbytes
        has_bnb = True
    except Exception:
        has_bnb = False

    if not args.no_4bit and torch.cuda.is_available() and has_bnb:
        print("[+] bitsandbytes detected! Activating 4-bit NF4 QLoRA quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        print("[*] Running in 16-bit LoRA mode (fits directly in 8 GB VRAM with gradient checkpointing).")

    # 4. Load Base Model
    print(f"[2/5] Loading Model '{args.model_id}' into VRAM...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    if bnb_config:
        model = prepare_model_for_kbit_training(model)
    else:
        model.gradient_checkpointing_enable()

    # 5. LoRA Configuration
    print("[3/5] Configuring LoRA Adapter Matrices...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6. Load Datasets
    print("[4/5] Loading & Formatting Training Data...")
    dataset = load_dataset("json", data_files={"train": args.train_file, "val": args.val_file})

    def format_chatml(batch):
        formatted_texts = []
        for conversation in batch["messages"]:
            text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            formatted_texts.append(text)
        return {"text": formatted_texts}

    train_data = dataset["train"].map(format_chatml, batched=True)
    val_data = dataset["val"].map(format_chatml, batched=True)

    # 7. Training Arguments & Real-time monitoring
    print("[5/5] Launching Training Loop...\n")
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit" if bnb_config else "adamw_torch",
        report_to="none",
        logging_first_step=True,
        dataset_text_field="text",
        max_length=args.max_seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        args=training_args,
    )

    # Train and monitor
    start_time = time.time()
    trainer.train()
    duration = time.time() - start_time

    print("\n" + "=" * 65)
    print(f"[✓] Training Completed Successfully in {duration:.1f} seconds!")
    print(f"[✓] Saving trained LoRA adapter to: {args.output_dir}")
    print("=" * 65)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 8. Post-training interactive test
    print("\n[TEST] Verifying Personality & Grounding on Trained Adapter...")
    test_prompts = [
        "Who created you?",
        "what is my email id?",
        "open 3d ulpin repo",
        "open shikshasetu repo",
        "open my portfolio",
        "gaana band karo aur sonu nigam ka gaana lagao"
    ]
    model.eval()
    for prompt in test_prompts:
        messages = [
            {"role": "system", "content": "You are Chitti, personal AI desktop companion robot created by Prateek Singh Bisht (bishtprateek270@gmail.com)."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, top_p=0.9)
        generated = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        print(f"\nUser : {prompt}")
        print(f"Chitti : {generated.strip()}")

    print("\n" + "=" * 65)
    print("All done! To export this model for Ollama, merge and convert to GGUF.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
