import numpy as np

import torch
import torch.nn as nn

# two layer convex CNN
class ConvexCNN(nn.Module):

    def __init__(self, input_shape, in_channels, out_channels, kernel_size, n_samples, n_rand):

        super(ConvexCNN, self).__init__()

        self.input_shape = input_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_samples = n_samples
        self.n_rand = n_rand

        # conv layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)      

        # randomly generated vectors to sample activation patterns
        self.u = torch.empty(n_rand, self.in_channels, self.kernel_size, self.kernel_size).normal_()

        # activation masks
        self.dmat = torch.empty(n_samples, n_rand, *(np.array(self.input_shape)-self.kernel_size+1), dtype=torch.bool)
        print(self.dmat.shape)

    # compute forward pass, need indices of data points since we use saved dmat
    def forward(self, X, i):
        
        return torch.sum(self.dmat[i] * self.conv1(X), dim=(-1,-2,-3))
        
    # compute masks from a portion of the data and add it to dmat
    def calc_masks(self, X, i):

        conv_res = torch.nn.functional.conv2d(X, self.u)

        self.dmat[i] = conv_res >= 0

    # filter dmat after populated based on the number of desired output channels
    def filter_masks(self):

        self.dmat, counts = torch.unique(self.dmat, dim=1, sorted=False, return_counts=True)

        idx = torch.topk(counts, k=self.out_channels, sorted=False).indices

        self.dmat = self.dmat[:, idx, :]
        print(self.dmat.shape)

# two layer relu CNN decomposed from convex CNN
class DecomposedConvexCNN(nn.Module):

    def __init__(self, convex_cnn: ConvexCNN):

        super(DecomposedConvexCNN, self).__init__()

        self.convex_cnn = convex_cnn

        # second conv layer for cone decomposition
        self.conv2 = nn.Conv2d(in_channels=convex_cnn.in_channels, out_channels=convex_cnn.out_channels, kernel_size=convex_cnn.kernel_size) 

    # loss function for approximate cone decomposition so that weights match activation patterns
    def decomposition_loss(self, X, i):
        
        X_u = self.convex_cnn.conv1(X)
        X_w = self.conv2(X)

        X_w_tilde = 2*self.convex_cnn.dmat[i]*X_w - X_w
        X_u_tilde = 2*self.convex_cnn.dmat[i]*X_u - X_u

        b = torch.nn.functional.relu(-X_u_tilde)
        return torch.norm(torch.nn.functional.relu(b - X_w_tilde))**2 + 1e-10*(torch.norm(self.conv2.weight)**2+torch.norm(self.conv2.bias)**2)
    
    # compute forward pass after decomposition
    def forward(self, X, i):

        conv_u = self.convex_cnn.conv1(X)
        conv_w = self.conv2(X)

        conv_v = conv_u + conv_w

        return torch.sum(torch.nn.functional.relu(conv_v) - torch.nn.functional.relu(conv_w), dim=(-1,-2,-3))