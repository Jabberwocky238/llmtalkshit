import torch
from safetensors import safe_open
from safetensors.torch import load_file
from safetensors.torch import save_file 
from LlamaForCausalLM.LlamaForCausalLM import LlamaForCausalLM
from LlamaForCausalLM.LlamaConfig import LlamaConfig
from transformers import AutoTokenizer

config_debug = LlamaConfig(
    hidden_size=256, # 2048
    intermediate_size=352, # 5632
    num_hidden_layers=4, # 22
    num_attention_heads=4, # 32
)
config = LlamaConfig()
model = LlamaForCausalLM(config_debug)

# safetensors = "../models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/77e23968eed12d195bd46c519aa679cc22a27ddc/model.safetensors"
# state_dict = load_file(safetensors)
# model.load_state_dict(state_dict, strict=False)

tokenizer_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir='../')

prompt = "How to get in a good university? Now I have 300 GRE scores and 100 TOEFL score. Will that gonna make me in Stanfold?"
formatted_prompt = (
    f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
)

tokens = tokenizer.encode(
    formatted_prompt,
    truncation=None,
    padding=False,
    max_length=None,
    add_special_tokens=False,
    return_tensors='pt',
)
# print(tokens)

model_outputs = model.generate(inputs=tokens, generation_config=config_debug, max_new_tokens=1024)
# print(model_outputs.size())

text = tokenizer.decode(
    model_outputs.tolist()[0],
    skip_special_tokens=True
)

print(text)