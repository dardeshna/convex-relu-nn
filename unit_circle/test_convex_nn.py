
import numpy as np
import torch

import matplotlib.pyplot as plt

import convex_nn
import two_layer_relu


def plot_decision(X, w1, w2, title):

    x_mesh, y_mesh = np.meshgrid(np.linspace(np.min(X[:,0])-0.2, np.max(X[:,0])+0.2, 100), np.linspace(np.min(X[:,1])-0.2, np.max(X[:,1])+0.2, 100))

    X_mesh = np.vstack((x_mesh.flatten(), y_mesh.flatten(), np.ones_like(x_mesh).flatten())).T
    z_mesh = (convex_nn.relu(X_mesh@w1)@w2).reshape(x_mesh.shape) > 0

    plt.figure(figsize=[5,5])
    plt.contourf(x_mesh, y_mesh, z_mesh, levels=1)
    plt.plot(X[y>0,0], X[y>0,1], 'bo')
    plt.plot(X[y<0,0], X[y<0,1], 'ro')

    t = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.sin(t), np.cos(t), ':k')
    plt.title(title)

    plt.savefig(title+'.pdf')


n = 50 # number of points
d = 3 # dimension of data including ones
m = 50 # number of neurons
beta = 1e-4

# generate data
np.random.seed(123)
X = np.random.randn(n,d-1)
X = np.append(X,np.ones((n,1)),axis=1)

y = ((np.linalg.norm(X[:,0:d-1],axis=1)>1)-0.5)*2

# construct loss function
loss_fn = two_layer_relu.get_hinge_loss(n, n, beta=beta)


# train convex nn
w1, w2 = convex_nn.train(X, y, m=50, beta=beta, oversample=True)

cvx_model = two_layer_relu.NonConvexRelu(d, m=50)

cvx_model.l1.weight.data = torch.Tensor(w1.T)
cvx_model.l2.weight.data = torch.Tensor(w2.T)

cvx_loss = loss_fn(cvx_model(torch.Tensor(X)).flatten(), torch.Tensor(y), cvx_model).item()
print("cvx model loss: ", cvx_loss)

plot_decision(X, w1, w2, title='binary_cvx')


# train nonconvex nn
losses, ncvx_model = two_layer_relu.train(X, y, m, beta, lr=1e-2)

ncvx_loss = loss_fn(ncvx_model(torch.Tensor(X)).flatten(), torch.Tensor(y), ncvx_model).item()
print("non-convex model loss: ", ncvx_loss)

w1 = ncvx_model.l1.weight.data.numpy().T
w2 = ncvx_model.l2.weight.data.numpy().T

plot_decision(X, w1, w2, title='binary_SGD')


# plot training loss
plt.figure()
plt.semilogy(losses, label='SGD')
plt.plot([0, len(losses)], [cvx_loss, cvx_loss], label='convex program')
plt.xlabel('epochs')
plt.ylabel('training loss')
plt.legend()
plt.savefig('binary_training_loss.pdf')

plt.show()