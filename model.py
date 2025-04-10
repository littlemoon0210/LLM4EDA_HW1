# llm_train/model.py
from transformers import GPT2Config, GPT2LMHeadModel

def get_model(
    n_embd=768,
    n_layer=12,
    n_head=12,
    vocab_size=50257  # 默认GPT2的vocab_size
):
    config = GPT2Config(
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        vocab_size=vocab_size
    )
    model = GPT2LMHeadModel(config)
    return model
