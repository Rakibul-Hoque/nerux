Nerux

A lightweight, NumPy-powered deep learning framework with a PyTorch-style API.

Nerux is a from-scratch neural network framework built entirely on top of NumPy. It implements automatic differentiation, tensors, layers, losses, optimizers, and data utilities with an API that will feel immediately familiar to anyone who has used PyTorch.

⚠️ Early Prototype — Nerux is under active development. The API is evolving, and some features may be incomplete or subject to change. Feedback, bug reports, and contributions are very welcome.

---

Table of Contents

· Why Nerux?
· Features
· Installation
· Quick Start
· Core Concepts
  · Tensors
  · Autograd
  · Disabling Gradients
· Tensor Operations
· Activations
· Layers
· Loss Functions
· Optimizers
· Learning Rate Schedulers
· Building Models
· Training Loop
· Data Utilities
· Transforms
· Saving, Loading & Evaluation
· End-to-End Examples
· How It Works
· Roadmap
· Contributing
· License

---

Why Nerux?

Most deep learning frameworks are massive, highly optimized C++/CUDA codebases. Nerux takes the opposite approach: it is a transparent, readable, NumPy-only implementation of the core ideas behind modern autodiff frameworks.

· Learn by reading — every gradient is computed with plain NumPy.
· PyTorch-like API — tensor, backward(), zero_grad(), optim.Adam, layers.Linear, Model, Dataset, DataLoader.
· Zero heavy dependencies — NumPy is all you need.
· Hackable — define custom layers, losses, optimizers, and transforms in a few lines.

If you want to understand how autograd, optimizers, and neural network layers actually work under the hood — Nerux is built for you.

---

Features

Category What's Included
Core tensor, broadcasting, autograd, dynamic computation graph
Math + - * / @ **, exp, log, pow, sum, mean, clip, argmax
Shape reshape, transpose, T, squeeze, unsqueeze, flatten
Activations relu, leakyrelu, sigmoid, tanh, softmax, softplus
Layers Linear, Dense, Conv2d, Embedding, LSTM, MultiHeadAttention, LayerNorm, Dropout, Sequential
Losses MSELoss, BCELoss, CrossEntropyLoss
Optimizers SGD, Adam (with weight_decay, grad_clip)
Schedulers Linear, CosineAnnealing
Data Dataset, DataLoader, transforms pipeline
Model API Model, forward, parameters(), save, evaluate, set_training

---

Installation

Nerux is currently distributed directly from GitHub.

Option 1 — Install via pip (recommended)

```bash
pip install git+https://github.com/Rakibul-Hoque/nerux.git
```

Option 2 — Clone and install in editable mode

```bash
git clone https://github.com/Rakibul-Hoque/nerux.git
cd nerux
pip install -e .
```

Option 3 — Clone and use directly

```bash
git clone https://github.com/Rakibul-Hoque/nerux.git
cd nerux
```

Then add the repository root to your PYTHONPATH, or run your scripts from within the project directory.

Requirements

· Python 3.8+
· NumPy

---

Quick Start

Train a linear regression model in under 30 lines:

```python
import nerux as nrx

# Parameters
w = nrx.tensor(0, requires_grad=True)
b = nrx.tensor(0, requires_grad=True)

# Model & loss (plain Python callables work fine)
model = lambda x: (x * w + b)
loss_fn = lambda yp, yt: ((yp - yt) ** 2).mean()

# Data: y = 2x + 1
x = nrx.tensor([2, 3, 4, 5])
y = nrx.tensor([5, 7, 9, 11])

lr = 0.01

for epoch in range(500):
    pred = model(x)
    loss = loss_fn(pred, y)

    loss.backward()

    with nrx.no_grad():
        w.sub_(w.grad * lr)
        b.sub_(b.grad * lr)

    w.zero_grad()
    b.zero_grad()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss = {loss.data}")

print("prediction:", model(x))
print("w =", w, "b =", b)
```

---

Core Concepts

Tensors

A tensor is an ndarray with autograd support.

```python
import nerux as nrx

# From Python scalars / lists
a = nrx.tensor(3.0)
b = nrx.tensor([1, 2, 3])
c = nrx.tensor([[1, 2], [3, 4]])

# With gradient tracking
w = nrx.tensor(0.0, requires_grad=True)

# Factory functions
z = nrx.zeros((2, 3))
o = nrx.ones((3, 3))
r = nrx.randn((3, 3), requires_grad=True)
u = nrx.uniform(-2, 2, (2, 5), requires_grad=True)
i = nrx.randint(0, 100, (100, 1, 28, 28))

# Inspect
print(a.data)     # raw NumPy value
print(a.shape)    # shape tuple
print(a.grad)     # gradient (None if not computed yet)
```

Autograd

Every operation on a tensor with requires_grad=True builds a dynamic computation graph. Call .backward() on any scalar output to populate .grad for every tensor that contributed to it.

```python
import nerux as nrx

x = nrx.randn((3, 3), requires_grad=True)
y = nrx.ones((3, 3))

z = ((x * y).exp() + x.pow(2) - y).mean()
z.backward()

print("z value:", z)
print("x grad:\n", x.grad)
```

Scalars broadcast against tensors just like in NumPy — and gradients flow back through them:

```python
x = nrx.randn((2, 3), requires_grad=True)
y = nrx(2.0, requires_grad=True)

z = (x * y + 3).mean()
z.backward()

print("x.grad =", x.grad)
print("y.grad =", y.grad)
```

Disabling Gradients

Use no_grad() to skip graph construction during manual parameter updates or inference:

```python
with nrx.no_grad():
    w.sub_(w.grad * lr)
    b.sub_(b.grad * lr)
```

You can also globally disable gradient tracking:

```python
nrx.stop_global_grad()
```

---

Tensor Operations

Nerux tensors behave like NumPy arrays, but track gradients.

```python
import nerux as nrx

x = nrx.randn((2, 3, 4))

print(x.transpose((1, 2, 0)).shape)   # (3, 4, 2)
print(x.clip(-1, 1).T)                # clipped + transposed
print(x.squeeze().reshape(6, 4))      # reshape
print(x.unsqueeze(0).shape)           # (1, 2, 3, 4)
print(x.mean().data)                  # scalar mean
```

Category Methods
Arithmetic +, -, *, /, @, **, sub_, add_, mul_, div_
Math exp, log, pow, sqrt, abs, clip
Reductions sum, mean, max, min, argmax
Shape reshape, transpose, T, squeeze, unsqueeze, flatten
Grad backward, zero_grad, requires_grad, grad, data

---

Activations

Activations are available both as tensor methods and as standalone layer classes.

As tensor methods:

```python
import nerux as nrx

a = nrx.randn((2, 3), requires_grad=True)

out = (
    a.relu()
     .tanh()
     .sigmoid()
     .softmax()
     .softplus()
     .leakyrelu(alpha=0.01)
     .sum()
)

out.backward()
print("a grad:\n", a.grad)
```

As layers:

```python
from nerux import layers

relu    = layers.ReLU()
sigmoid = layers.Sigmoid()
softmax = layers.Softmax()
```

Available: ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, Softplus.

---

Layers

All layers live in nerux.layers.

Linear / Dense

```python
from nerux import layers

linear = layers.Linear(1)          # output features
linear.input(2)                    # input features

x = linear(nrx.randn((4, 2)))
print(x.shape)
```

Dense combines a linear projection with an activation:

```python
from nerux import layers

fc = layers.Dense(64, activation=layers.ReLU())
```

Convolutional Layers

```python
from nerux import layers

conv = layers.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
pool = layers.MaxPool2d(kernel_size=2)
flat = layers.Flatten()
```

Recurrent Layers

```python
from nerux import layers

lstm = layers.LSTM(hidden_size=64, num_layers=2, dropout=0.4)
out, (h_n, c_n) = lstm(x)
```

Embedding & Attention

```python
from nerux import layers

emb = layers.Embedding(num_embeddings=5003, embedding_dim=64, padding_idx=0)
mha = layers.MultiHeadAttention(d_model=64, num_heads=4, dropout=0.1)
```

Normalization & Regularization

```python
from nerux import layers

norm    = layers.LayerNorm(64)
dropout = layers.Dropout(0.1)
```

Sequential

Stack layers into a single module:

```python
from nerux import layers

block = layers.Sequential([
    layers.Dense(128, activation=layers.ReLU()),
    layers.Dropout(0.2),
    layers.Dense(10, activation=layers.Softmax()),
])
```

Custom Layers

Subclass layers.Layer and implement forward:

```python
from nerux import layers

class ResidualBlock(layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = layers.Dense(dim, activation=layers.ReLU())
        self.fc2 = layers.Dense(dim)
        self.norm = layers.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.fc2(self.fc1(x)))
```

---

Loss Functions

Losses live in nerux.losses.

```python
from nerux import losses

mse  = losses.MSELoss()
bce  = losses.BCELoss()
ce   = losses.CrossEntropyLoss()

loss = bce(pred, target)
loss.backward()
```

---

Optimizers

Optimizers live in nerux.optim.

```python
from nerux import optim

optimizer = optim.Adam(model.parameters(), lr=0.001)
# or
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

Standard training step:

```python
optimizer.zero_grad()   # clear old gradients
loss.backward()         # compute new gradients
optimizer.step()        # update parameters
```

Options:

Argument Description
lr Learning rate
weight_decay L2 regularization
grad_clip Gradient clipping (max norm)

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.002,
    weight_decay=1e-4,
    grad_clip=1.0,
)
```

---

Learning Rate Schedulers

Schedulers live in nerux.schedulers.

```python
from nerux import schedulers

# Linear decay
scheduler = schedulers.Linear(
    optimizer,
    start_factor=1.0,
    end_factor=0.5,
    total_epochs=15,
)

# Cosine annealing
scheduler = schedulers.CosineAnnealing(optimizer, T_max=20, min_lr=1e-5)
```

Call scheduler.step() once per epoch.

---

Building Models

Subclass Model for a PyTorch-like module hierarchy.

```python
import nerux as nrx
from nerux import Model, layers


class SubModel(layers.Layer):
    def __init__(self):
        super().__init__()
        self.ln1   = layers.Linear(60)
        self.relu1 = layers.ReLU()
        self.ln2   = layers.Linear(60)
        self.relu2 = layers.ReLU()

    def forward(self, x):
        return self.relu2(self.ln2(self.relu1(self.ln1(x))))


class MLP(Model):
    def __init__(self):
        super().__init__()
        self.ln1 = layers.Dense(50, activation=layers.ReLU())
        self.ln2 = SubModel()
        self.ln3 = layers.Dense(4, activation=layers.Softmax())
        self.input(2)                 # declare input dimensionality

    def forward(self, x):
        return self.ln3(self.ln2(self.ln1(x)))


model = MLP()
print(model.parameters())             # list of trainable tensors
```

A Minimal Linear Model

```python
import nerux as nrx
from nerux import layers, losses, optim

model = layers.Linear(1)
model.input(2)

x = nrx.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], requires_grad=True)
y = nrx.tensor([[1], [0], [0], [1]], requires_grad=True)

loss_fn   = losses.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}: loss = {loss.data}")

print("prediction:", model(x))
```

---

Training Loop

A complete, canonical Nerux training loop:

```python
import nerux as nrx
from nerux import optim, losses, layers, Model
from nerux.data import Dataset, DataLoader


class MLP(Model):
    def __init__(self):
        super().__init__()
        self.ln1 = layers.Dense(10, activation=layers.ReLU())
        self.ln2 = layers.Dense(16, activation=layers.ReLU())
        self.ln3 = layers.Dense(4,  activation=layers.Softmax())
        self.input(2)

    def forward(self, x):
        return self.ln3(self.ln2(self.ln1(x)))


model = MLP()

x = nrx.randn((100, 2))
y = nrx.randn((100, 4))

dataset = Dataset(x, y)
loader  = DataLoader(dataset, batch_size=10)

loss_fn   = losses.MSELoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(100):
    running = 0.0
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += loss.data

    print(f"Epoch {epoch}: loss = {running / len(loader):.4f}")
```

With Validation & Early Stopping

```python
model.set_training(True)     # training mode (dropout on)
model.set_training(False)    # evaluation mode (dropout off)

best_val, patience, wait = float("inf"), 4, 0

for epoch in range(15):
    model.set_training(True)
    train_loss = 0.0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data

    model.set_training(False)
    val_loss = 0.0
    for xb, yb in val_loader:
        val_loss += loss_fn(model(xb), yb).data

    t = train_loss / len(train_loader)
    v = val_loss   / len(val_loader)
    scheduler.step()

    print(f"epoch {epoch + 1} train:{t:.4f}, val:{v:.4f}")

    if v < best_val - 1e-4:
        best_val, wait = v, 0
        model.save("model.nrx")
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stop at epoch {epoch + 1}")
            break
```

---

Data Utilities

Nerux ships a lightweight Dataset / DataLoader pair in nerux.data.

```python
import numpy as np
import nerux as nrx
from nerux.data import Dataset, DataLoader

x = np.load("x_train.npy")
y = np.load("y_train.npy")

split = int(len(x) * 0.8)
train_ds = Dataset(x[:split], y[:split])
val_ds   = Dataset(x[split:], y[split:])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

for xb, yb in train_loader:
    print(xb.shape, yb.shape)
```

Feature Description
Dataset(x, y) Pair inputs with targets
DataLoader(ds, batch_size, shuffle) Iterate mini-batches
dataset.split(ratio=0.8) Train/val split
dataset[i] Fetch a single transformed sample

---

Transforms

nerux.data.transforms provides a composable preprocessing pipeline.

```python
import nerux as nrx
from nerux.data import Dataset, transforms

x = nrx.randint(0, 100, (100, 1, 28, 28))
y = nrx.randint(0, 9, (100, 1))


class CustomTransform(transforms.Custom):
    def __init__(self):
        self.reshape = transforms.Reshape(28, 1, 28)

    def __call__(self, x):
        return self.reshape(x)


standardScaler = transforms.StandardScaler().fit(x)
minMaxScaler   = transforms.MinMaxScaler().fit(x)

transform = transforms.Compose([
    standardScaler.transform,
    minMaxScaler.transform,
    CustomTransform(),
    transforms.Reshape(28, 28, 1),
    transforms.Flatten(),
    transforms.Normalize(mean=10, std=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomNoise(std=0.01),
])

dataset = Dataset(
    x, y,
    transform=transform,
    target_transform=transforms.OneHotEncoder(num_classes=10),
)
```

Available transforms: Compose, Reshape, Flatten, Normalize, StandardScaler, MinMaxScaler, RandomHorizontalFlip, RandomNoise, OneHotEncoder, Custom.

---

Saving, Loading & Evaluation

```python
# Save the best checkpoint
model.save("model.nrx")

# Evaluate on a dataset — writes metrics to JSON
print(model.evaluate(test_dataset, file="evl.json"))
print(model.evaluate(train_dataset))

# Export / import parameters as plain dicts
params = model.parameters_dict()
with open("model_params.json", "w") as f:
    json.dump(params, f, indent=2)

model.set_parameters(params)
```

---

End-to-End Examples

1. Manual Autograd — Two-Layer Network

Build a fully-connected network from raw tensors, no layers required:

```python
import nerux as nrx
from nerux import optim, losses

W1 = nrx.uniform(-2, 2, (2, 5), requires_grad=True)
B1 = nrx.zeros((5,), requires_grad=True)
W2 = nrx.uniform(-2, 2, (5, 1), requires_grad=True)
B2 = nrx.zeros((1,), requires_grad=True)


def linear(x):
    return ((((x @ W1) + B1).relu() @ W2) + B2).sigmoid()


x = nrx.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y = nrx.tensor([[1], [0], [0], [1]])

loss_fn   = losses.BCELoss()
optimizer = optim.Adam([W1, B1, W2, B2], lr=0.01)

for epoch in range(500):
    pred = linear(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss = {loss.data}")

print("prediction:", linear(x))
```

This is the XOR problem — solved with pure Nerux.

2. CNN Image Classification

```python
import nerux as nrx
from nerux import Model, layers, losses, optim
from nerux.data import Dataset, DataLoader


class CNN(Model):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = layers.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = layers.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = layers.MaxPool2d(2)
        self.relu  = layers.ReLU()
        self.flat  = layers.Flatten()
        self.fc1   = layers.Dense(128, activation=layers.ReLU())
        self.drop  = layers.Dropout(0.3)
        self.fc2   = layers.Dense(num_classes, activation=layers.Softmax())
        self.input(1, 28, 28)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flat(x)
        x = self.drop(self.fc1(x))
        return self.fc2(x)


model     = CNN()
loss_fn   = losses.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# x: (N, 1, 28, 28)  y: (N, 10) one-hot
train_loader = DataLoader(Dataset(x_train, y_train), batch_size=64, shuffle=True)

for epoch in range(10):
    model.set_training(True)
    total = 0.0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.data

    print(f"Epoch {epoch}: loss = {total / len(train_loader):.4f}")
```

3. LSTM Text Classification

```python
import numpy as np
from nerux import Model, layers, losses, optim, schedulers
from nerux.data import Dataset, DataLoader


class LSTMModel(Model):
    def __init__(self, vocab, input_size, hidden_size, dropout=0.4, num_layers=2):
        super().__init__()
        self.emb      = layers.Embedding(vocab, input_size, padding_idx=0)
        self.drop_emb = layers.Dropout(dropout)
        self.lstm     = layers.LSTM(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.drop_out = layers.Dropout(dropout)
        self.fc       = layers.Dense(1, activation=layers.Sigmoid())

    def forward(self, x):
        x = self.emb(x)
        x = self.drop_emb(x)
        out, (h_n, c_n) = self.lstm(x)
        mean = out.mean(axis=1)
        return self.fc(self.drop_out(mean))


x = np.load("x_train.npy")
y = np.load("y_train.npy")

split = int(len(x) * 0.8)
train_ds = Dataset(x[:split], y[:split])
val_ds   = Dataset(x[split:], y[split:])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

sequence_len = 50
model = LSTMModel(vocab=5003, input_size=64, hidden_size=64).input(sequence_len)

loss_fn   = losses.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4, grad_clip=1.0)
scheduler = schedulers.Linear(optimizer, start_factor=1.0, end_factor=0.5, total_epochs=15)

best_val, patience, wait = float("inf"), 4, 0

for epoch in range(15):
    model.set_training(True)
    train_loss = 0.0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data

    model.set_training(False)
    val_loss = 0.0
    for xb, yb in val_loader:
        val_loss += loss_fn(model(xb), yb).data

    t = train_loss / len(train_loader)
    v = val_loss   / len(val_loader)
    scheduler.step()

    print(f"epoch {epoch + 1} train:{t:.4f}, val:{v:.4f}")

    if v < best_val - 1e-4:
        best_val, wait = v, 0
        model.save("model.nrx")
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stop at epoch {epoch + 1}")
            break
```

4. Transformer Encoder Classifier

```python
from nerux import Model, layers
from nerux.data import Dataset, DataLoader


class CausalSelfAttention(layers.Layer):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.mha     = layers.MultiHeadAttention(d_model, num_heads, dropout)
        self.norm    = layers.LayerNorm(d_model)
        self.dropout = layers.Dropout(dropout)

    def forward(self, x):
        attn = self.dropout(self.mha(x, x, x, None))
        return self.norm(x + attn)          # residual connection


class FeedForward(layers.Layer):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1     = layers.Dense(d_ff, activation=layers.ReLU())
        self.fc2     = layers.Dense(d_model)
        self.norm    = layers.LayerNorm(d_model)
        self.dropout = layers.Dropout(dropout)

    def forward(self, x):
        ff = self.dropout(self.fc2(self.fc1(x)))
        return self.norm(x + ff)            # residual connection


class EncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = CausalSelfAttention(d_model, num_heads, dropout)
        self.ff        = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        return self.ff(self.self_attn(x))


class TransformerClassifier(Model):
    def __init__(self, vocab, d_model=64, num_heads=4,
                 seq_len=50, num_layers=2, dropout=0.1):
        super().__init__()
        self.token_emb = layers.Embedding(vocab, d_model)
        self.encoders  = layers.Sequential([
            EncoderBlock(d_model, num_heads, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        self.fc = layers.Dense(1, activation=layers.Sigmoid())

    def forward(self, x):
        x = self.token_emb(x)
        x = self.encoders(x)
        x = x.mean(axis=1)
        return self.fc(x)
```

---

How It Works

Nerux implements reverse-mode automatic differentiation on top of NumPy.

1. Forward pass — every operation on a tensor with requires_grad=True records itself and its inputs into a dynamic computation graph.
2. Backward pass — calling .backward() traverses the graph in reverse topological order, applying the chain rule and accumulating gradients into .grad.
3. Parameter update — optimizers read .grad and apply update rules (SGD, Adam, ...), then zero_grad() clears the buffers for the next iteration.

No C extensions. No CUDA. Just NumPy — which makes every gradient easy to read, debug, and learn from.

---

Roadmap

Nerux is an early prototype. Planned and in-progress work includes:


· More optimizers (RMSProp, AdamW, SGD with Nesterov)
· More schedulers (StepLR, ExponentialLR, Warmup)
· Full CNN support (strided/padded convs, BatchNorm, pooling variants) 
· Positional encodings & full Transformer decoder blocks
· Model checkpoint format & serialization improvements
· Expanded test suite and benchmarking
· Documentation site

Have a feature request? Open an issue on GitHub.

---

Contributing

Contributions are warmly welcomed — this project is in its early stages, which makes it a great time to get involved.

```bash
# 1. Fork the repository
# 2. Clone your fork
git clone https://github.com/<your-username>/nerux.git
cd nerux

# 3. Create a feature branch
git checkout -b feature/my-awesome-feature

# 4. Install in editable mode
pip install -e .

# 5. Make your changes, then commit
git commit -m "Add my awesome feature"

# 6. Push and open a Pull Request
git push origin feature/my-awesome-feature
```

Guidelines

· Keep the API consistent with PyTorch conventions where possible.
· Add tests for new operators, layers, and optimizers.
· Prefer clarity over micro-optimization — Nerux is a learning-first framework.
· Update the README if you add user-facing features.

Found a bug? Please open an issue with a minimal reproduction.

---

License

This project is licensed under the terms of the license included in the repository. See LICENSE for details.

---

Acknowledgements

Nerux is heavily inspired by PyTorch — its API design, module system, and autograd semantics. It is powered entirely by NumPy.

If Nerux helped you learn something, consider giving the repo a ⭐ — it genuinely helps.

⬆ Back to top