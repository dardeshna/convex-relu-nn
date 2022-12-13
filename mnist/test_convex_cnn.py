import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader

from convex_cnn_pytorch import ConvexCNN, DecomposedConvexCNN

class MNISTWrapper(Dataset):

    def __init__(self, **kwargs):

        mnist = datasets.MNIST(**kwargs)
        mask = ((mnist.targets == 0) + (mnist.targets == 1)).nonzero().view(-1)
        self.subset = Subset(mnist,mask)
        
    def __getitem__(self, index):
        data, target = self.subset[index]
        return index, data, target

    def __len__(self):
        return len(self.subset)

    def __getattr__(self, attr):
        return getattr(self.subset, attr)


train_data = MNISTWrapper(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
test_data = MNISTWrapper(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)

mask_loader = DataLoader(train_data, batch_size=1000, shuffle=False)
train_loader = DataLoader(train_data, batch_size=100, shuffle=False)
test_loader = DataLoader(test_data, batch_size=100, shuffle=False)

# create and train the convex model
def train(num_epochs=10, lr=1e-3, n_rand=32, m=16, k=3):

    n_batches = len(train_loader)
    n_samples = len(train_data)
    
    model = ConvexCNN((28,28), 1, m, k, n_samples, n_rand)
    loss_func = nn.BCEWithLogitsLoss()

    for i, (idx, images, labels) in enumerate(mask_loader):

        model.calc_masks(images, idx)

        print(f"batch {i+1} / {len(mask_loader)}")
    
    print("Calculated masks")

    model.filter_masks()

    print("Filtered masks")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    losses = []

    for epoch in range(num_epochs):

        samples = 0
        accum = 0

        for i, (idx, images, labels) in enumerate(train_loader):

            samples += len(labels)

            output = model(images, idx)       

            loss = loss_func(output, labels.float())
            
            # backpropagation, compute gradients 
            loss.backward()

            accum += loss.item()
            
            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, n_batches, accum/samples))
        
            # apply gradients             
            optimizer.step()

            # clear gradients for this training step   
            optimizer.zero_grad()

        losses.append(accum/samples)

    return losses, model

# perform cone decomposition
def decomp(convex_cnn, num_epochs=10, lr=1e-2):

    n_batches = len(train_loader)

    model = DecomposedConvexCNN(convex_cnn)

    optimizer = torch.optim.Adam(model.conv2.parameters(), lr=lr)
        
    losses = []

    for epoch in range(num_epochs):

        samples = 0
        accum = 0

        for i, (idx, images, labels) in enumerate(train_loader):

            samples += len(labels)

            loss = model.decomposition_loss(images, idx)

            # backpropagation, compute gradients 
            loss.backward()

            accum += loss.item()
            
            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, n_batches, accum/samples))

            # apply gradients             
            optimizer.step()

            # clear gradients for this training step   
            optimizer.zero_grad()

        losses.append(accum/samples)

    return losses, model

# evaluate accuracy both models on the training data
def eval_train(model, decomp_model):

    # Test the model
    model.eval()
    decomp_model.eval()

    with torch.no_grad():

        model_correct = 0
        decomp_model_correct = 0

        squared_errors = []

        total = 0

        for idx, images, labels in train_loader:

            total += len(labels)

            model_output = model(images, idx)
            decomp_model_output = decomp_model(images, idx)

            squared_errors.append(torch.norm(model_output-decomp_model_output)**2)

            model_pred_y = model_output >= 0
            decomp_model_pred_y = decomp_model_output >= 0

            model_correct += (model_pred_y == labels).sum().item()
            decomp_model_correct += (decomp_model_pred_y == labels).sum().item()
        
        model_accuracy = model_correct / total
        print('Training Accuracy of the model: %.2f' % model_accuracy)

        decomp_model_accuracy = decomp_model_correct / total
        print('Training Accuracy of the decomp model: %.2f' % decomp_model_accuracy)

        print("MSE between models:", np.sum(squared_errors)/total)
    
# evaluate accuracy of the decomposed model on the test data
def test(decomp_model):

    # Test the model
    decomp_model.eval()

    with torch.no_grad():

        model_correct = 0
        total = 0

        for idx, images, labels in test_loader:

            total += len(labels)

            output = decomp_model(images, idx)

            pred_y = output >= 0

            model_correct += (pred_y == labels).sum().item()
        
        model_accuracy = model_correct / total
        print('Test Accuracy of the model: %.2f' % model_accuracy)


if __name__ == "__main__":

    losses, model = train()
    # torch.save(model, 'convex_model.pt')

    plt.semilogy(losses)
    plt.title('training loss')
    plt.xlabel('epochs')
    plt.ylabel('BCE loss')
    # plt.savefig('mnist_training_loss.pdf')

    # model = torch.load('convex_model.pt')

    decomp_losses, decomp_model = decomp(model)
    # torch.save(decomp_model, 'decomp_model.pt')

    plt.figure()
    plt.semilogy(decomp_losses)
    plt.title('decomp loss')
    plt.xlabel('epochs')
    plt.ylabel('MSE loss')
    # plt.savefig('mnist_decomp_loss.pdf')

    # decomp_model = torch.load('decomp_model.pt')

    eval_train(model, decomp_model)
    test(decomp_model)

    plt.show()