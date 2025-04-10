# llm_train/train.py
from model import get_model
from dataset import get_dataset_and_tokenizer
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import os
import wandb

# === 初始化 wandb 项目 ===
wandb.init(
    project="gpt2-small-train",
    name="bookcorpus-bsz2-epoch3",
    config={
        "model": "GPT2-small",
        "dataset": "bookcorpus",
        "epochs": 3,
        "batch_size": 2
    }
)

# === 检查 checkpoint ===
print("🟢 Start training script")
if os.path.exists("./logs/checkpoint-last"):
    print("✅ Found checkpoint: logs/checkpoint-last. Will resume from checkpoint.")
else:
    print("🚀 No checkpoint found. Training will start from scratch.")

# === 加载数据和分词器 ===
tokenizer, train_dataset, val_dataset = get_dataset_and_tokenizer(dataset_name="bookcorpus")

# === 加载模型 ===
model = get_model(vocab_size=tokenizer.vocab_size)

# === 训练参数 ===
training_args = TrainingArguments(
    output_dir="./logs",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    fp16=True,
    report_to="wandb",
)

# === 开始训练 ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.train(resume_from_checkpoint=True)

# === 保存模型 ===
trainer.save_model("./logs/final_model")
print("✅ Training complete! Model saved to ./logs/final_model")
