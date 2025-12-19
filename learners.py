import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

