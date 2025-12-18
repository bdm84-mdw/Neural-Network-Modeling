import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as transforms

def gen_nonlinear_data(num_samples=10000):
    # generate random x samples for training and test sets
    xTr = torch.rand(num_samples, 1) * 2 * np.pi
    xTe = torch.rand(int(num_samples * 0.1), 1) * 2 * np.pi
    
    # gaussian noise for non-linear regression
    noise = torch.rand(num_samples, 1) * 0.2
    test_noise = torch.rand(int(num_samples * 0.1), 1) * 0.2
    
    # add noise on the labels for the training set
    yTr = torch.sin(xTr) + noise
    yTe = torch.sin(xTe) + test_noise
    return xTr, xTe, yTr, yTe

nl_xTr, nl_xTe, nl_yTr, nl_yTe = gen_nonlinear_data(num_samples=500)

class LinearRegressionModel(nn.Module):
    def __init__(self, ndims):
        super(LinearRegressionModel, self).__init__()
        """ pytorch optimizer checks for the properties of the model, and if
            the torch.nn.Parameter requires gradient, then the model will update
            the parameters automatically.
        """
        self.w = nn.Parameter(torch.randn(ndims, 1), requires_grad=True)
        self.b = nn.Parameter(torch.randn(1), requires_grad=True)
    
    def forward(self, x):
        return x @ self.w + self.b
    
def mse_loss(y_pred, y_true):
    square_diff = torch.pow((y_pred-y_true), 2)
    mean_error = 0.5 * torch.mean(square_diff)
    return mean_error

# Create a custom SGD optimizer with momentum
class CustomSGD(optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super(CustomSGD, self).__init__(params, defaults)
        self.velocities = [torch.zeros_like(param.data) for param in self.param_groups[0]['params']]
    
    def step(self):
        for group in self.param_groups:
            for param, velocity in zip(group['params'], self.velocities):
                if param.grad is None:
                    continue
                
                lr = group['lr'] # learning rate
                momentum = group['momentum'] # momentum coefficient
                gradient = param.grad.data # gradient
                
                # update the velocity; [:] enables inplace update
                velocity[:] = momentum * velocity + (1-momentum)*gradient
                
                # update the parameters
                param.data = param.data - lr*velocity

def train_regression_model(xTr, yTr, model, num_epochs, lr=1e-2, momentum=0.9, print_freq=100, display_loss=True):
    """Train loop for a neural network model.
    
    Input:
        xTr:     (n, d) matrix of regression input data
        yTr:     n-dimensional vector of regression labels
        model:   nn.Model to be trained
        num_epochs: number of epochs to train the model for
        lr:      learning rate for the optimizer
        print_freq: frequency to display the loss
        display_loss: boolean, if we print the loss
    
    Output:
        model:   nn.Module trained model
    """
    optimizer = CustomSGD(model.parameters(), lr=lr, momentum=momentum)  # create an Adam optimizer for the model parameters
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()  
        pred = model(xTr)  # run the forward pass through the model to compute predictions
        loss = mse_loss(pred, yTr)
        loss.backward()  # compute the gradient wrt loss
        optimizer.step()  # performs a step of gradient descent
        if display_loss and (epoch + 1) % print_freq == 0:
            print('epoch {} loss {}'.format(epoch+1, loss.item()))
    
    return model  # return trained model

ndims = nl_xTr.shape[1]
linear_model = LinearRegressionModel(ndims)  # initialize the model
linear_model = train_regression_model(nl_xTr, nl_yTr,linear_model, num_epochs=2000, lr=1e-2, momentum=0.9, print_freq=500)
avg_test_error = mse_loss(linear_model(nl_xTe), nl_yTe)  # compute the average test error
print('linear regression avg test error', avg_test_error.item())

# Visualize the results
plt.figure()
plt.plot(nl_xTr, linear_model(nl_xTr).detach(),linewidth=5.0, color="red", label="Prediction Line")
plt.scatter(nl_xTr, nl_yTr, label="Train Points")
plt.scatter(nl_xTe, nl_yTe, label="Test Points")
plt.legend()
plt.show()

# Create a Pytorch Multilayer Perceptron (MLP) Model
class MLPNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(MLPNet, self).__init__()
        """ pytorch optimizer checks for the properties of the model, and if
            the torch.nn.Parameter requires gradient, then the model will update
            the parameters automatically.
        """
        self.input_dim = input_dim
        # Initialize the fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
            
    
    def forward(self, x):
        # Implement the forward pass, with ReLU non-linearities
        y = x.view(-1,self.input_dim)
        return self.fc2(nn.functional.relu(self.fc1(y)))

hdims = 90
num_epochs = 5000
lr = 1e-1
momentum = 0.9

mlp_model = MLPNet(input_dim=1, hidden_dim=hdims, output_dim=1)
mlp_model = train_regression_model(nl_xTr, nl_yTr, mlp_model, num_epochs=num_epochs, lr=lr, momentum=momentum)
avg_test_error = mse_loss(mlp_model(nl_xTe), nl_yTe)
print('Feedforward neural network avg test error', avg_test_error.item())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3))

# Plot the visualizations from our MLP Model
ax1.scatter(nl_xTr, nl_yTr, label="Train Points")
ax1.scatter(nl_xTe, nl_yTe, label="Test Points")
ax1.scatter(nl_xTr, mlp_model(nl_xTr).detach(), color="red", marker='o', label="Prediction")
ax1.legend()
ax1.set_title('MLP Net')

# Plot the visualizations from our MLP Model
ax2.scatter(nl_xTr, nl_yTr, label="Train Points")
ax2.scatter(nl_xTe, nl_yTe, label="Test Points")
ax2.plot(nl_xTr, linear_model(nl_xTr).detach(),linewidth=5.0, color="red", label="Prediction Line")
ax2.legend()
ax2.set_title('Linear Model')

plt.show()



###################################################
"""MNIST Classification by Convolutional Neural Network"""

# Load the dataset
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=".", train=True, transform=trans, download=True)
test_set = dset.MNIST(root=".", train=False, transform=trans, download=True)

batch_size = 64

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

# Training function
def train_classification_model(train_loader, model, num_epochs, lr=1e-1, momentum=0.9, print_freq=100):
    """Train loop for a neural network model.
    
    Input:
        train_loader:    Data loader for the train set. 
                         Enumerate through to train with each batch.
        model:           nn.Model to be trained
        num_epochs:      number of epochs to train the model for
        lr:              learning rate for the optimizer
        print_freq:      frequency to display the loss
    
    Output:
        model:   nn.Module trained model
    """
    optimizer = CustomSGD(model.parameters(), lr=lr, momentum=momentum)  # create an SGD optimizer for the model parameters
    for epoch in range(num_epochs):
        # Iterate through the dataloader for each epoch
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            # imgs (torch.Tensor):    batch of input images
            # labels (torch.Tensor):  batch labels corresponding to the inputs
            optimizer.zero_grad()
            predictions = model(imgs)
            loss = nn.functional.cross_entropy(predictions, labels)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % print_freq == 0:
            print('epoch {} loss {}'.format(epoch+1, loss.item()))
            # Implement the training loop using imgs, labels, and cross entropy loss   
    return model  # return trained model

# Test function
def test_classification_model(test_loader, model):
    """Tests the accuracy of the model.
    
    Input:
        test_loader:      Data loader for the test set. 
                          Enumerate through to test each example.
        model:            nn.Module model being evaluate.
        
    Output:
        accuracy:         Accuracy of the model on the test set.
    """
    # Compute the model accuracy
    
    temp = torch.tensor([])
    for batch_idx, (imgs, labels) in enumerate(test_loader):
        predictions = torch.argmax(model(imgs),dim = 1)
        acc = (predictions == labels).float()
        temp = torch.cat((temp,acc))
    accuracy = torch.mean(temp)
    
    #batches shouldn't be equally weighted!
    return accuracy

# Train an MLP model on MNIST
hidden_dim = 50
num_epochs = 40
lr = 1e-2
momentum = 0.9

mlp_model = MLPNet(input_dim=(28*28), hidden_dim=hidden_dim, output_dim=10)
print('the number of parameters', sum(parameter.view(-1).size()[0] for parameter in mlp_model.parameters()))
mlp_model = train_classification_model(train_loader, mlp_model, num_epochs=num_epochs, lr=lr, momentum=momentum)
avg_test_acc = test_classification_model(test_loader, mlp_model)
print('avg test accuracy', avg_test_acc)

# Create Pytorch ConvNet Model
class ConvNet(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=32, output_dim=1):
        super(ConvNet, self).__init__()
        
        # Initialize the ConvNet layers
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, 3, stride = 1, padding = 1, bias = False)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, stride = 1, padding = 1, bias = False)
        
        self.fc = nn.Linear(49*hidden_channels,output_dim)
        
        
    def forward(self, x):
        # Implement the forward pass, with ReLU non-linearities and max-pooling
        fst = nn.functional.max_pool2d(nn.functional.relu(self.conv1(x)), kernel_size = 2, stride = 2)
        snd = nn.functional.max_pool2d(nn.functional.relu(self.conv2(fst)), kernel_size = 2, stride = 2)
        
        return self.fc(snd.view(-1,self.fc.in_features))

# Train a convnet model on MNIST
hidden_channels = 20
num_epochs = 40
lr = 1e-3
momentum = 0.9

conv_model = ConvNet(input_channels=1, hidden_channels=hidden_channels, output_dim=10)
print('the number of parameters:', sum(parameter.view(-1).size()[0] for parameter in conv_model.parameters()))
conv_model = train_classification_model(train_loader, conv_model, num_epochs=num_epochs, lr=lr, momentum=momentum, print_freq=1)
avg_test_acc = test_classification_model(test_loader, conv_model)
print('avg test accuracy', avg_test_acc)