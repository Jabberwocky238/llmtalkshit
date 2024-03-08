import torch
from safetensors import safe_open
from safetensors.torch import load_file
from safetensors.torch import save_file 
from LlamaForCausalLM.LlamaForCausalLM import LlamaForCausalLM
from LlamaForCausalLM.LlamaConfig import LlamaConfig

safetensors = "../models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/77e23968eed12d195bd46c519aa679cc22a27ddc/model.safetensors"
state_dict = load_file(safetensors)
config = LlamaConfig()
model = LlamaForCausalLM(config)
model.load_state_dict(state_dict, strict=False)

from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, TextGenerationPipeline
import transformers 
import torch

# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# model = LlamaForCausalLM.from_pretrained(model_name, cache_dir='../')
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir='../')

# print(model)
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )
pipeline = TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    # device_map="auto"
)

prompt = "How to get in a good university?"
formatted_prompt = (
    f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
)


sequences = pipeline(
    formatted_prompt,
    do_sample=True,
    top_k=50,
    top_p = 0.9,
    num_return_sequences=1,
    repetition_penalty=1.1,
    max_new_tokens=1024,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")