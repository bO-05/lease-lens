"""
finetune_legal.py  ·  Lease Lens / AutoScientist shared model core
===================================================================
QLoRA fine-tune of an 8-9B model on the CUAD legal-clause dataset, on Modal,
with a one-liner GGUF export for local (llama.cpp / Ollama / MLX) serving.

This produces the ONE model you wrap twice:
  • HF Build Small  -> load the GGUF locally inside the Lease Lens Gradio app
  • AutoScientist   -> push merged weights + the dataset to HF and Kaggle

------------------------------------------------------------------
RUN IT
------------------------------------------------------------------
  pip install modal
  modal token new                      # one-time auth
  modal secret create huggingface HF_TOKEN=hf_xxx     # your HF write token
  modal run finetune_legal.py                          # full run (~1 epoch)
  modal run finetune_legal.py --smoke-test             # 60 steps, ~5 min sanity check

Cost/time on an A100-40GB (~$2.10/hr on Modal): a 1-epoch run is roughly
30-90 min => ~$1-3 of your $250 credits. Iterate freely.

NOTE: defaults to Llama-3.1-8B (battle-tested, definitely on Unsloth). To use the
current best ~8-9B at kickoff (e.g. a Qwen3.5-9B Unsloth 4-bit build), just swap
BASE_MODEL below and re-run.
"""

import modal

# --- config ---------------------------------------------------------------
BASE_MODEL   = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"   # 3B: Tiny Titan class
HF_REPO      = "giladam01/lease-lens-legal-3b-v25"             # Modal-trained, 20% negatives
MAX_SEQ_LEN  = 2048          # matches our proven 3B recipe
DATASET      = "chenghao/cuad_qa"                              # parquet mirror of CUAD (script-free; loads on new datasets)
GGUF_QUANT   = "q4_k_m"      # ~4.7GB for 8B, runs on a 16GB Mac

SYSTEM_PROMPT = (
    "You are a meticulous legal contract analyst. Given a contract excerpt and a "
    "clause category, extract the exact verbatim text of any clause that matches "
    "that category. If no matching clause is present, reply with exactly: NONE."
)

# --- Modal app & image -----------------------------------------------------
app = modal.App("lease-lens-3b-v25")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "cmake", "git", "curl")   # needed for GGUF/llama.cpp build
    .pip_install(
        "unsloth", "trl", "peft", "accelerate",
        "bitsandbytes", "datasets", "huggingface_hub", "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

vol = modal.Volume.from_name("legal-ft", create_if_missing=True)


# --- dataset formatting ----------------------------------------------------
def build_dataset(tokenizer, smoke_test: bool):
    """Positives (answer-windowed) + ~20% synthesized NONE negatives (absent categories)."""
    import collections, random
    from datasets import load_dataset, Dataset

    split = "train[:400]" if smoke_test else "train"
    ds = load_dataset(DATASET, split=split)

    def wrap(q):
        return "Highlight any part of this contract related to: " + q + ". If there is none, reply NONE."

    def to_text(q, ctx, target):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": wrap(q) + "\n\n---\nContract:\n" + ctx},
            {"role": "assistant", "content": target},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def window(context, answers):
        starts = answers.get("answer_start") or []
        if starts:
            s = max(0, starts[0] - 1000)
            return context[s:s + 4000]
        return context[:4000]

    pos, by_title, ctx_of = [], collections.defaultdict(set), {}
    for row in ds:
        ans = row.get("answers", {})
        texts = ans.get("text") or []
        target = " | ".join(dict.fromkeys(t.strip() for t in texts if t.strip())) or "NONE"
        pos.append(to_text(row["question"], window(row["context"], ans), target))
        by_title[row["title"]].add(row["question"])
        ctx_of.setdefault(row["title"], row["context"])

    all_cats = set().union(*by_title.values()) if by_title else set()
    rng = random.Random(3407)
    neg = []
    for t, covered in by_title.items():
        absent = sorted(all_cats - covered)
        rng.shuffle(absent)
        for cat in absent[:8]:
            neg.append(to_text(cat, ctx_of[t][:4000], "NONE"))
    rng.shuffle(neg)
    neg = neg[:int(0.25 * len(pos))]          # 20% of total
    texts = pos + neg
    rng.shuffle(texts)
    print(f"mix: {len(pos)} positives + {len(neg)} negatives ({100*len(neg)/max(len(texts),1):.0f}% NONE)")
    return Dataset.from_dict({"text": texts})

# --- training function -----------------------------------------------------
@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 60 * 3,
    volumes={"/outputs": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def train(smoke_test: bool = False, push: bool = True):
    import os, torch
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    # 1) load base in 4-bit
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,
    )

    # 2) attach LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # 3) data
    dataset = build_dataset(tokenizer, smoke_test)
    print(f"Training examples: {len(dataset)}")

    # 4) train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LEN,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,      # effective batch 16
            warmup_steps=10,
            max_steps=(60 if smoke_test else 450),
            learning_rate=2e-4,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="/outputs/checkpoints",
            report_to="none",
        ),
    )
    trainer.train()

    # 5) save merged 16-bit weights (for the AutoScientist HF/Kaggle release)
    model.save_pretrained_merged("/outputs/merged-16bit", tokenizer, save_method="merged_16bit")

    vol.commit()

    # 6) push the ADAPTER to the Hub (small + fast).
    #    NOTE: Unsloth's in-job GGUF export needs an interactive apt-get prompt that
    #    Modal containers can't answer (caused the smoke-test crash) -- so GGUF for
    #    v2.5 is produced later with the proven no-build Colab converter if needed.
    if push and os.environ.get("HF_TOKEN"):
        model.push_to_hub(HF_REPO, token=os.environ["HF_TOKEN"])
        tokenizer.push_to_hub(HF_REPO, token=os.environ["HF_TOKEN"])
        print(f"Adapter pushed to https://huggingface.co/{HF_REPO}")

    print("Done. Artifacts in the 'legal-ft' volume: /outputs/merged-16bit and /outputs/gguf")


@app.local_entrypoint()
def main(smoke_test: bool = False, push: bool = True):
    train.remote(smoke_test=smoke_test, push=push)
