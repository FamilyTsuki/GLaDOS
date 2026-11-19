# GLaDOS LoRA fine-tune of Qwen3-4B-Instruct-2507 on Google Colab (free T4 GPU).
# Paste each "# %%" cell into its own Colab cell, in order.
# Input : glados_dataset_final.jsonl (upload it in the Colab file panel)
# Output: glados-q4_k_m.gguf to import into Ollama (see training/Modelfile.example)

# %% 1. Install
# !pip install -q unsloth

# %% 2. Load the base model in 4-bit (same model as Ollama's qwen3:4b-instruct)
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3-4B-Instruct-2507", max_seq_length=2048, load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=42,
)

# %% 3. Dataset -> chat-template text (json loaded by hand: mixed tool/no-tool messages break Arrow schemas)
import json
from datasets import Dataset

rows = [json.loads(l) for l in open("glados_dataset_final.jsonl", encoding="utf-8")]
texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False) for r in rows]
# Unsloth ships the hybrid Qwen3 template, which adds an empty <think> block to the last reply;
# Instruct-2507 (and Ollama's qwen3:4b-instruct) never think, so drop it.
texts = [t.replace("<think>\n\n</think>\n\n", "") for t in texts]
ds = Dataset.from_dict({"text": texts}).train_test_split(test_size=0.05, seed=42)
print(len(ds["train"]), "train /", len(ds["test"]), "eval")
print(texts[0])

# %% 4. Train (loss only on GLaDOS's replies)
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=ds["train"], eval_dataset=ds["test"],
    args=SFTConfig(
        dataset_text_field="text", per_device_train_batch_size=4, gradient_accumulation_steps=4,
        packing=False,  # Unsloth packs by default: 1681 short examples became 88 rows = only 12 steps
        num_train_epochs=2, learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=10,
        logging_steps=10, eval_strategy="epoch", optim="adamw_8bit", weight_decay=0.01,
        seed=42, output_dir="outputs", report_to="none",
    ),
)
trainer = train_on_responses_only(
    trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n",
)
trainer.train()

# %% 5. Quick test
FastLanguageModel.for_inference(model)
for q in ["Comment faire cuire des pâtes ?", "What's the capital of Australia?", "Tu es nulle."]:
    ids = tokenizer.apply_chat_template([{"role": "user", "content": q}],
                                        add_generation_prompt=True, return_tensors="pt").to("cuda")
    out = model.generate(input_ids=ids, max_new_tokens=120, temperature=0.7, top_p=0.8)
    print(q, "->", tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True), "\n")

# %% 6. Export GGUF Q4_K_M (~2.5 GB), then download it from the file panel
model.save_pretrained_gguf("glados", tokenizer, quantization_method="q4_k_m")
# !ls -lh glados*  # the .gguf file to download
