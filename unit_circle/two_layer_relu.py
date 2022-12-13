import numpy as np

import torch
from torch import nn

def relu(x):
    return np.maximum(0,x)

# simple two-layer relu network
class NonConvexRelu(nn.Module):
    def __init__(self, d, m):
        super(NonConvexRelu, self).__init__()
        self.l1 = nn.Linear(d, m, bias=False)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(m, 1, bias=False)

    def forward(self, x):
        return self.l2(self.a1(self.l1(x)))

# build a hinge loss function
def get_hinge_loss(N_train, batch_size, beta):

    def hinge_loss(pred, y, model):

        w1 = model.l1.weight.data
        w2 = model.l2.weight.data

        return torch.sum(torch.nn.functional.relu(1-y*pred))/N_train + beta*batch_size/(2*N_train)*(torch.sum(w1**2) + torch.sum(w2**2))

    return hinge_loss

# build a squared loss function
def get_squared_loss(N_train, batch_size, beta):

    def squared_loss(pred, y, model):

        w1 = model.l1.weight.data
        w2 = model.l2.weight.data

        return 1/2*torch.sum((y-pred)**2) + beta*batch_size/(2*N_train)*(torch.sum(w1**2) + torch.sum(w2**2))

    return squared_loss

# create and train a nonconvex model
def train(X, y, m=50, beta=1e-4, lr=1e-3, batch_size=25, num_iters=10000):
    
    N_train = X.shape[0]

    model = NonConvexRelu(X.shape[1], m)

    loss_fn = get_hinge_loss(N_train, batch_size, beta)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    X = torch.Tensor(X)
    y = torch.Tensor(y)

    losses = []

    num_batches = (N_train+batch_size-1)//batch_size

    for i in range(num_iters):

        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)

        X = X[idx]
        y = y[idx]

        accum = 0

        for j in range(num_batches):
            
            start = j*batch_size
            end = min((j+1)*batch_size, N_train)

            X_ = X[start:end]
            y_ = y[start:end]

            pred = model(X_).flatten()

            loss = loss_fn(pred, y_, model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accum += loss.item()

        losses.append(accum)

        if i % 100 == 0:
            print(f"loss: {accum:>7f}  [{i:>5d}/{num_iters:>5d}]")

    return losses, model
