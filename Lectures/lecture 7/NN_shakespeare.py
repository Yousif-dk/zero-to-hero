import torch
import torch.nn as nn
from torch.nn import functional as F

#Hyperparameters
#-----------------------------
batch_size = 64 #how many independent sequences will we process in parallel?
block_size = 256 #maximum context length for predictions?
max_iters = 5000
eval_interval = 500
lr = 3e-4
device = 'cuda' if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384 #number of embedding dimensions
n_layer = 6
n_head = 6
dropout = 0.2
#-----------------------------
print(device)
#-----------------------------
torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#here are all the unique chars that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

#create a mapping from chars to ints:
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] #encoder! feed it a string, shits out a vector of ints
decode = lambda d: ''.join([itos[c] for c in d]) #decoder! feed it a vector of ints, shits out a string

#train & test splits:
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#data loader:
def get_batch(split):
    #generate a small batch of data inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()

def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """one head of self attention!"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        #Compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 ## (B,T,C) @ # (B,C,T) => (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        #Perform the weighted aggregration of the values
        v = self.value(x) # (B,T,C)
        out = wei@ v # (B,T,T) @ (B,T,C) => (B,T,C) 
        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""
    def __init__(self,num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        return out
    

class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    """transformer block: communication (Multi-head attention) followed by computation (feedforwards)"""
    def __init__(self, n_embd, n_head):
        #n_embd = embedding dimension, n_head: the number of heads we want 
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self,x):
        x = x + self.sa(self.ln1(x)) #Residual connections for stability, this boy is becoming big.
        x = x + self.ffwd(self.ln2(x)) #Residual connections for stability, this boy is becoming big.
        return x
#super simple bigram model (UPDATE:  we done threw in some head.)

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        #self.blocks = nn.Sequential(
            #Block(n_embd, n_head=4),
            # Block(n_embd, n_head=4),
            #  Block(n_embd, n_head=4),
            #   nn.LayerNorm(n_embd))
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) #final layer norm
        #to go from token embeddings to logits we need a linear layer: 
        self.lm_head = nn.Linear(n_embd, vocab_size) #short for language modelling head
        #self.sa_heads = MultiHeadAttention(4, n_embd//4)
        #self.ffwd =FeedForward(n_embd)


    def forward(self,idx,targets = None):
        #idx and targets are both (B,T) tensors of integers:
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx) #(B,T,C)
        # ^^^^^Batch_size = 4, Time = block_size = 8, channel = vocab_size = 65^^^^^
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) #(T,C)
        x = tok_emb + pos_emb #(B,T,C)
        #x = self.sa_heads(x) #apply multiple heads of self-attention (B,T,C)
        #x = self.ffwd(x) #(B,T,C)
        x = self.blocks(x) #(B,T,C)
        x = self.ln_f(x) #(B,T,C)
        logits = self.lm_head(x) #(B,T,vocab_size)


        if targets == None:
            loss = None
        else:
            #loss2 = F.cross_entropy(logits,targets) #this by itself doesn't work. cross_entropy expects B C T, not B T C
            B, T , C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop idx to the last blocl_size tokens:
            idx_cond = idx[:, -block_size:]
            #get the predictions
            logits, loss = self(idx_cond)
            #focus only on the last time step
            logits = logits[:,-1,:] #Becomes (B,C)
            #apply softmax to get probabilities
            probs = F.softmax(logits, dim=1) #(B,C)
            #sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            #append sampled index to the running sequence
            idx = torch.cat((idx,idx_next), dim = 1) #(B, T+1)
        return idx

model = BigramLanguageModel()

m = model.to(device)
print(sum(p.numel() for p in m.parameters() if p.requires_grad), "parameters")
print("--------------------")
print("Parameter breakdown:")
for name, param in m.named_parameters():
    if param.requires_grad:
        print(f"{name:30s} {param.numel():>10,d}")
print("Total parameters:", sum(p.numel() for p in m.parameters() if p.requires_grad))

#create a Pytorch optimizer. 

optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

for iter in range(max_iters):
    #every once in a while evaluate the loss on train and val loss
    print(iter)
    if iter%eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train_loss {losses['train']:.4f}, val_loss: {losses['val']:.4f}")
    
    #sample a batch of data
    xb, yb = get_batch("train")

    #evaluate the loss 
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate from the model:
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
