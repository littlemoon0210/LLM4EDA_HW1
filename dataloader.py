# llm_train/dataset.py
from datasets import load_dataset
from transformers import GPT2Tokenizer
import os

# 设置数据集缓存路径
os.environ["HF_DATASETS_CACHE"] = "/share/home/kexiaoyue/Desktop/LLM4EDA/hw1_llm_trainning/bookcorpus/"

def get_dataset_and_tokenizer(dataset_name="bookcorpus", val_split=0.05, block_size=128, debug=False):
    tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_tokenizer")
    tokenizer.pad_token = tokenizer.eos_token

    # 加载原始数据集（从缓存路径）
    dataset = load_dataset(dataset_name, split="train")

    # 分词函数
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            return_special_tokens_mask=True,
            truncation=True,
            max_length=512
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        load_from_cache_file=False,
        num_proc=os.cpu_count(),
        desc="Tokenizing dataset"
    )

    #  添加映射后检查函数
    if debug:
        sample = tokenized[0]
        print("[Debug] Tokenized sample keys:", sample.keys())
        print("[Debug] input_ids length:", len(sample["input_ids"]))
        print("[Debug] special_tokens_mask length:", len(sample["special_tokens_mask"]))
        assert len(sample["input_ids"]) == len(sample["special_tokens_mask"]), "input_ids 和 special_tokens_mask 长度不一致"

    # 划分训练集和验证集
    split = tokenized.train_test_split(test_size=val_split, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]

    # 切块函数：将 input_ids 切成 block_size 的一组组
    def group_texts(examples):
        concatenated = sum(examples["input_ids"], [])
        total_len = (len(concatenated) // block_size) * block_size
        return {
            "input_ids": [concatenated[i:i + block_size] for i in range(0, total_len, block_size)]
        }

    train_dataset = train_dataset.map(
        group_texts,
        batched=True,
        num_proc=os.cpu_count(),
        desc="Grouping train set"
    )
    val_dataset = val_dataset.map(
        group_texts,
        batched=True,
        num_proc=os.cpu_count(),
        desc="Grouping val set"
    )

    if debug:
        print("Grouped train sample:", train_dataset[0])
        print("HF_DATASETS_CACHE:", os.environ.get("HF_DATASETS_CACHE"))

    return tokenizer, train_dataset, val_dataset
