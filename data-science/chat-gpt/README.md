# ChatGPT

## Step by step

### Collab Preparation

#### Load dataset
```sh
## Load the dataset
!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```
```python
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
```
```python
print("length of dataset in characters: ", len(text))
```
```python
## let's look at the first 1000 characters
print(text[:1000])
```

#### Tokenization, train/val split
```python
## here are all the unique characters that ocurr in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print("length of vocabulary: ", vocab_size)
```
```python
# [DEMO]: Encoder and Decoder to create a integer sequence based in our vocabulary
## create a mapping from characters to integers  
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a lista of integers
decode = lambda l: ''.join([itos[c] for c in l]) # decoder: take a lista of integers, output a string

print("encoded: ", encode("ola pessoas!"))
print("decoded: ", decode(encode("ola pessoas!")))
```
```python
## let's now encode the entire text dataset and store it into a torch.Tensor
import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000]) # the characters we looke at earlier to the GPT look like this
```
```python
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
```
#### Data loader
> Let's load chunks of data from the dataset to our model, not all at once because computational resources are limited and it's expensive.

```python
block_size = 8
## in a chunk of 9 charactecters, there's actually 8 indivudal examples packed there
### In a context of 18, next is 47,
### In a context of 18, 46 next is 56... 
train_data[:block_size+1]
```
```python
x = train_data[:block_size] # inputs to the transformer
y = train_data[1:block_size+1] # next block size characters (offset by 1)
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target is {target}")
# > transformer will never receive more than block size and after block size we need to start trunctating
```
```python
torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will be process in parallel?
block_size = 8 # what is the maximum contextl ength for predictions?

def get_batch(split):
    # generate a small batch of data of inputs "X" and targets "Y"
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # random offsets on the training set
    x = torch.stack([data[i:i+block_size] for i in ix]) # inputs to the transformer
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # next block size characters (offset by 1)
    return x, y

xb, yb = get_batch('train')
print("inputs:")
print(xb.shape)
print(xb)
print("/\ each of theses are a chunk of the training set 4x8")
print("targets:")
print(yb.shape)
print(yb)

print('----') 

for b in range(batch_size): #batch dimension
    for t in range(block_size): #time dimension
        context = xb[b,:t+1]
        target = yb[b,t]
        print(f"when input is {context} the target is {target}")
```

#### simplest baseline: bigram language model, loss, generation

> [Collab Link](https://colab.research.google.com/drive/1ZdWs9gti7teonEjzu52haNxojF8qAs1j?usp=sharing)

## Questions?
- What is `torch.Tensor`?
  - ???

## Tools

- [Google Collab](https://colab.research.google.com/)

## References

- [Let's build GPT: from scratch, in code, spelled out.](https://youtu.be/kCc8FmEb1nY)