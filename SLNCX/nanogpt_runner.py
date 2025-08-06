import json
import os
import pickle
from contextlib import nullcontext

import torch
import tiktoken

from nanogpt_model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume'
out_dir = 'out'
start = "\n"
num_samples = 1
max_new_tokens = 100
temperature = 0.8
top_k = 200
seed = 1337
device = 'cpu'
dtype = 'float32'
compile = False
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.to(device)
if compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']

    def encode(s: str) -> list[int]:
        return [stoi[c] for c in s]

    def decode(ids: list[int]) -> str:
        return ''.join([itos[i] for i in ids])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")

    def encode(s: str) -> list[int]:
        return enc.encode(s, allowed_special={"<|endoftext|>"})

    def decode(ids: list[int]) -> str:
        return enc.decode(ids)

def _load_tokens(path: str) -> torch.Tensor:
    tokens = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            tokens.extend(encode(obj["content"]))
    return torch.tensor(tokens, dtype=torch.long)


def fine_tune(dataset_dir: str, steps: int = 100, lr: float = 3e-4) -> None:
    train_path = os.path.join(dataset_dir, "train.jsonl")
    tokens = _load_tokens(train_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    block_size = getattr(gptconf, "block_size", 128)
    model.train()
    for _ in range(steps):
        ix = torch.randint(0, len(tokens) - block_size - 1, (1,))
        x = tokens[ix : ix + block_size].to(device)[None, ...]
        y = tokens[ix + 1 : ix + 1 + block_size].to(device)[None, ...]
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "model_args": gptconf.__dict__,
        "config": {"dataset": dataset_dir},
    }, os.path.join(out_dir, "ckpt.pt"))


def generate_text() -> None:
    model.eval()
    # encode the beginning of the prompt
    prompt = start
    if prompt.startswith('FILE:'):
        with open(prompt[5:], 'r', encoding='utf-8') as f:
            prompt = f.read()
    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    with torch.no_grad():
        with ctx:
            for _ in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                print(decode(y[0].tolist()))
                print('---------------')


dataset = globals().get('dataset')
if dataset:
    fine_tune(dataset)
else:
    generate_text()
