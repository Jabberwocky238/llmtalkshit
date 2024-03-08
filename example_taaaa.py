# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# import fire

from llama import Llama
from typing import List

from dotenv import load_dotenv
load_dotenv()

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128, # The maximum sequence length for input prompts
    max_gen_len: int = 64, # The maximum length of generated sequences
    max_batch_size: int = 4,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """ 
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompts: List[str] = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        # "我认为你像个傻逼",
        # "Simply put, the theory of relativity states that ",
        # """A brief message congratulating the team on the launch:

        # Hi everyone,
        
        # I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        # """Translate English to French:
        
        # sea otter => loutre de mer
        # peppermint => menthe poivrée
        # plush girafe => girafe peluche
        # cheese =>""",
    ]
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        # print(prompt)
        print("==================================")
        print(f"{result['generation']}")
        print("==================================")


if __name__ == "__main__":
    ckpt_dir = '../model--llama-2-7b/'
    tokenizer_path = "../model--Llama-2-70b-chat-tokenizer/tokenizer.model"
    main(ckpt_dir, tokenizer_path)
