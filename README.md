# PyTorch Basics: Understanding the Training Loop

This repository contains a **simple PyTorch example** designed to demonstrate the **core fundamentals of model training**, with a particular focus on the canonical PyTorch training loop:

```python
for epoch in range(epochs):
    # Forward pass
    y_hat = model(X)

    loss = loss_fn(y_hat, y_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
