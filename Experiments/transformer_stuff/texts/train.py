# train.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from model import TransformerLanguageModel, TransformerConfig

# ---------------- Hyperparameters ----------------
#Must Follow Rules:
#n_embd % n_head == 0
#--------------------------------------------------
batch_size   = 64
block_size   = 256
max_iters    = 5000
eval_interval = 500
lr           = 3e-4
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
eval_iters   = 200
n_embd       = 200
n_layer      = 1
n_head       = 1
dropout      = 0.2
# -------------------------------------------------
print("We running this bad boy on:")
print(device)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda d: ''.join([itos[c] for c in d])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

def get_batch(split):
    src = train_data if split == 'train' else val_data
    ix = torch.randint(len(src) - block_size, (batch_size,))
    x = torch.stack([src[i:i+block_size] for i in ix])
    y = torch.stack([src[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out

# Build model from config (no globals in model.py)
cfg = TransformerConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout,
)
model = TransformerLanguageModel(cfg).to(device)

# Parameter report (optional)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_params:,} parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {iter}: train_loss {losses['train']:.4f}, val_loss {losses['val']:.4f}")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

ckpt = {
    "model_state": model.state_dict(),
    "config": cfg.__dict__,          # so we can rebuild the model
    "itos": itos,                    # decode ints -> chars
    "stoi": stoi,                    # encode chars -> ints
}
torch.save(ckpt, "texts_transformer.pt")
print("Saved to texts_transformer.pt")
