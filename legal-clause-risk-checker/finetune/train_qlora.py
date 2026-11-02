"""QLoRA fine-tuning of Qwen/Qwen2.5-1.5B-Instruct for clause classification.

Fine-tunes Qwen2.5-1.5B-Instruct on CUAD clause classification data using
4-bit QLoRA. The trained LoRA adapter is saved to finetune/adapters/.

Hyperparameter choices (explained inline):
  - r=32, alpha=64: LoRA rank/scaling. Higher rank gives the adapters more
    expressivity to capture legal-language patterns; alpha=64 (2x r) is a
    common stable ratio.
  - target_modules: all linear layers in the transformer block
    (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj) so the
    adapters can modify both attention and MLP projections.
  - batch_size=1, gradient_accumulation_steps=16: effective batch of 16,
    which fits in limited VRAM while still providing stable gradient updates.
  - learning_rate=2e-4: standard QLoRA learning rate, high enough for the
    adapters to adapt quickly but stable enough not to destroy the base model.
  - epochs=3: enough passes over the small clause dataset without overfitting.
  - gradient_checkpointing=True: trades a small compute cost for much lower
    memory usage, essential on consumer GPUs.
  - bf16=True: bfloat16 mixed precision — natively supported on RTX 40-series
    (Ada Lovelace) and more numerically stable than fp16.
  - max_seq_length=1024: fits typical clause lengths while bounding memory.
  - load_in_4bit=True (strict): 4-bit NF4 quantization via bitsandbytes,
    reducing memory footprint ~4x vs full precision to fit a 1.5B model on
    a single 6GB consumer GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
ADAPTERS_DIR = PROJECT_ROOT / "finetune" / "adapters"

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 1024

# LoRA target modules for Qwen2.5 (same naming convention as LLaMA-style
# transformers: attention projections + MLP projections).
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_prompt(text: str) -> str:
    """Format a clause as a classification prompt for the model."""
    return f"Clause: {text}\nLabel: "


def tokenize_example(
    text: str, label: str, tokenizer, max_length: int
) -> dict[str, list[int]]:
    """Tokenize a (prompt, label) pair with input masking.

    Only the label portion contributes to the loss; the prompt portion is
    masked with -100 so the model learns to predict the label given the
    clause text.
    """
    prompt = format_prompt(text)
    full_text = prompt + label

    full_enc = tokenizer(
        full_text, truncation=True, max_length=max_length, padding=False
    )
    prompt_enc = tokenizer(
        prompt, truncation=True, max_length=max_length, padding=False
    )
    prompt_len = len(prompt_enc["input_ids"])

    input_ids = full_enc["input_ids"][:max_length]
    labels = [-100] * prompt_len + full_enc["input_ids"][prompt_len:]
    labels = labels[:max_length]
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def evaluate_accuracy(
    model, tokenizer, val_records: list[dict], max_seq_length: int
) -> float:
    """Evaluate classification accuracy on the held-out validation set.

    Generates the label for each clause and compares to the ground truth
    (case-insensitive exact match). Runs on the model's device (CUDA).
    """
    import torch

    model.eval()
    correct = 0
    total = 0
    device = model.device

    for record in val_records:
        prompt = format_prompt(record["text"])
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=32,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (after the prompt).
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()

        if prediction.lower() == record["label"].lower():
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def train(
    model_name: str = DEFAULT_MODEL,
    train_path: Path | None = None,
    val_path: Path | None = None,
    output_dir: Path | None = None,
    epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 2e-4,
    max_seq_length: int = MAX_SEQ_LENGTH,
    max_samples: int | None = None,
) -> None:
    """Run QLoRA fine-tuning on CUDA with strict 4-bit NF4 quantization.

    Heavy imports (torch, transformers, peft, bitsandbytes, datasets) are
    local so the module imports cleanly without them installed.

    Requires CUDA: 4-bit NF4 quantization via bitsandbytes is enforced
    (load_in_4bit=True). bf16 is used for mixed precision, which is natively
    supported on RTX 40-series (Ada Lovelace) GPUs and more stable than fp16.
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for QLoRA fine-tuning. "
            "4-bit NF4 quantization via bitsandbytes requires an NVIDIA GPU."
        )

    train_path = train_path or (DATA_PROCESSED / "clause_classification_train.jsonl")
    val_path = val_path or (DATA_PROCESSED / "clause_classification_val.jsonl")
    output_dir = output_dir or ADAPTERS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data -------------------------------------------------------
    train_records = load_jsonl(train_path)
    val_records = load_jsonl(val_path)

    if max_samples is not None:
        train_records = train_records[:max_samples]
        val_records = val_records[: max(1, max_samples // 10)]

    print(
        f"Train records: {len(train_records)} | Val records: {len(val_records)}"
    )

    # ---- Tokenizer -------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize all examples up front.
    tokenized_train = [
        tokenize_example(r["text"], r["label"], tokenizer, max_seq_length)
        for r in train_records
    ]
    tokenized_val = [
        tokenize_example(r["text"], r["label"], tokenizer, max_seq_length)
        for r in val_records
    ]

    train_dataset = Dataset.from_list(tokenized_train)
    val_dataset = Dataset.from_list(tokenized_val)

    # ---- Model: strict 4-bit NF4 quantization (CUDA only) ----------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    print("Loaded model with strict 4-bit NF4 quantization (CUDA).")

    # ---- LoRA ------------------------------------------------------------
    lora_config = LoraConfig(
        r=32,                          # LoRA rank
        lora_alpha=64,                 # LoRA scaling (alpha = 2 * r)
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Training arguments ----------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,   # 1: small batch for memory
        gradient_accumulation_steps=gradient_accumulation_steps,  # 16: eff. batch 16
        learning_rate=learning_rate,              # 2e-4: standard QLoRA LR
        num_train_epochs=epochs,                  # 3: passes over data
        gradient_checkpointing=True,              # save memory
        bf16=True,                                # bfloat16 on RTX 40-series
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    # ---- Trainer ---------------------------------------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # ---- Save adapter ----------------------------------------------------
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Adapter saved to {output_dir}")

    # ---- Validation accuracy ---------------------------------------------
    accuracy = evaluate_accuracy(model, tokenizer, val_records, max_seq_length)
    print(f"Validation accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune Qwen2.5-1.5B-Instruct for clause classification"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--val", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=16
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit training samples (quick smoke test)",
    )
    args = parser.parse_args()

    train(
        model_name=args.model,
        train_path=args.train,
        val_path=args.val,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        max_samples=args.max_samples,
    )
