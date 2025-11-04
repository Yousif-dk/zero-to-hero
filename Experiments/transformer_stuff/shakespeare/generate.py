import torch
from model import TransformerLanguageModel, TransformerConfig
import sys, time

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    cfg = TransformerConfig(**ckpt["config"])
    model = TransformerLanguageModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["stoi"], ckpt["itos"]

def encode(s, stoi):
    return torch.tensor([[stoi[c] for c in s]], dtype=torch.long)

def decode(ids, itos):
    return "".join(itos[i] for i in ids)

@torch.no_grad()
def stream_generate(model, stoi, itos, prompt="", max_new_tokens=500,
                    temperature=1.0, top_k=None, device="cpu", delay=0.0):
    if prompt:
        idx = encode(prompt, stoi).to(device)
        print(prompt, end="", flush=True)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(1e-8, temperature)

        if top_k is not None:
            v, ix = torch.topk(logits, top_k)
            mask = torch.full_like(logits, float("-inf"))
            logits = mask.scatter(1, ix, v)

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

        token = itos[next_id.item()]
        print(token, end="", flush=True)
        if delay > 0:
            time.sleep(delay)

    print()  # newline at end

if __name__ == "__main__":
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Running generation on {device}...\n")

    model, stoi, itos = load_checkpoint("shakespeare_transformer.pt", device)

    prompt = ""
    max_new_tokens = 1000
    temperature = 0.9
    top_k = 50

    print("Generating text...\n")
    stream_generate(
        model, stoi, itos,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        device=device,
        delay=0.01,  # ⏱️ adjust this to slow down output (0 for fastest)
    )
