# llm_train/dataset.py
from datasets import load_dataset
from transformers import GPT2Tokenizer

def get_dataset_and_tokenizer(dataset_name="bookcorpus", val_split=0.05, block_size=128):
    # 加载 GPT2 的分词器
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 加载数据集
    dataset = load_dataset(dataset_name, split="train")

    # 分词函数
    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 划分训练集和验证集
    split = tokenized.train_test_split(test_size=val_split, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]

    # 切块函数
    def group_texts(examples):
        concatenated = sum(examples["input_ids"], [])
        total_len = (len(concatenated) // block_size) * block_size
        return {
            "input_ids": [concatenated[i:i + block_size] for i in range(0, total_len, block_size)]
        }

    train_dataset = train_dataset.map(group_texts, batched=True)
    val_dataset = val_dataset.map(group_texts, batched=True)

    return tokenizer, train_dataset, val_dataset
