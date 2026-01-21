# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 2026

@author: antal
"""

import torch 
import torch.nn as nn

N = 10
D_in = 1
D_out = 1
X = torch.randn(N, D_in)

true_W = torch.tensor([[2.0]])
true_b = torch.tensor(1.0)

y_true = X@true_W + true_b + torch.randn(N,D_out)*0.1

# Hyperparameters
learning_rate, epochs = 0.01, 100

# R
W, b = torch.randn(1, 1, requires_grad = True), torch.randn(1, requires_grad=True)

# Training loop
for epoch in range(epochs):
    # fore
    y_hat = X @ W + b
    loss = torch.mean((y_hat - y_true)**2)
    
    loss.backward()
    
    with torch.no_grad():
        W -= learning_rate * W.grad; b -= learning_rate * b.grad
        
    W.grad.zero_(); b.grad.zero_()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d}: Loss={loss.item():.4f}")
        
        
class LinearRegressionModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear_layer(x)
    
model = LinearRegressionModel(in_features = 1 , out_features = 1)

import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

for epoch in range(epochs):
    
    #forward pass
    y_hat = model(X)
    
    loss = loss_fn(y_hat, y_true)
    
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
