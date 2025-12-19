import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms

import learners

"""MNIST Classification by Convolutional Neural Network"""
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 

def data_prep():
    # Load the dataset
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = dset.MNIST(root=".", train=True, transform=trans, download=True)
    test_set = dset.MNIST(root=".", train=False, transform=trans, download=True)

    batch_size = 64

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    return train_loader,test_loader

train_loader, test_loader = data_prep()

# Training function
def train_classification_model(train_loader, model, num_epochs, lr=1e-1, momentum=0.9, print_freq=1):
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
    optimizer = learners.CustomSGD(model.parameters(), lr=lr, momentum=momentum)  # create an SGD optimizer for the model parameters
    for epoch in range(num_epochs):
        # Iterate through the dataloader for each epoch
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            # imgs (torch.Tensor):    batch of input images
            # labels (torch.Tensor):  batch labels corresponding to the inputs
            imgs, labels = imgs.to(device), labels.to(device)
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
    temp = torch.tensor([]).to(device)
    for batch_idx, (imgs, labels) in enumerate(test_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        predictions = torch.argmax(model(imgs),dim = 1)
        acc = (predictions == labels).float()
        temp = torch.cat((temp,acc))
    accuracy = torch.mean(temp)
    
    #batches shouldn't be equally weighted!
    return accuracy

# Train and evaluate an MLP model on MNIST
def eval_FNN(hidden_dim, num_epochs, lr, momentum):
    mlp_model = learners.MLPNet(input_dim=(28*28), hidden_dim=hidden_dim, output_dim=10)
    print('the number of parameters', sum(parameter.view(-1).size()[0] for parameter in mlp_model.parameters()))
    mlp_model = mlp_model.to(device)
    mlp_model = train_classification_model(train_loader, mlp_model, num_epochs=num_epochs, lr=lr, momentum=momentum)
    avg_test_acc = test_classification_model(test_loader, mlp_model)
    print('avg test accuracy', avg_test_acc)
    return mlp_model, avg_test_acc

# Train and evaluate a convnet model on MNIST
def eval_CNN(hidden_channels, num_epochs, lr, momentum):
    conv_model = learners.ConvNet(input_channels=1, hidden_channels=hidden_channels, output_dim=10)
    print('the number of parameters:', sum(parameter.view(-1).size()[0] for parameter in conv_model.parameters()))
    conv_model = conv_model.to(device)
    conv_model = train_classification_model(train_loader, conv_model, num_epochs=num_epochs, lr=lr, momentum=momentum, print_freq=1)
    avg_test_acc = test_classification_model(test_loader, conv_model)
    print('avg test accuracy', avg_test_acc)
    return conv_model, avg_test_acc

##################################################
"""Demo of CNN and FNN classification"""
def imshow(img):
    img = img.detach().cpu()
    img = img + 0.5     # unnormalize
    plt.figure()
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.show()

def cnn_fnn_demo(fnn, cnn):
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    # Make a forward pass to get predictions of the MLP model
    mlp_scores = fnn(images)
    _, mlp_preds = torch.max(mlp_scores.data, 1)

    # Make a forward pass to get predictions of the ConvNet model
    conv_scores = cnn(images)
    _, conv_preds = torch.max(conv_scores.data, 1)

    show_img_idx = np.random.randint(images.shape[0], size=7)
    # show images
    imshow(torchvision.utils.make_grid(images[show_img_idx]))
    # print labels
    print('labels are:', ' '.join('%d' % labels[j] for j in show_img_idx))
    # print predictions
    print('MLP predictions are:', ' '.join('%d' % mlp_preds[j] for j in show_img_idx))
    print('CNN predictions are:', ' '.join('%d' % conv_preds[j] for j in show_img_idx))

##################################################
"""Hidden dimension optimization"""
def display_result(x, y, title, xlabel, ylabel):
    """Displays the plot of y vs. x given title, xlabel, ylabel"""
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x)
    plt.show()

def parameter_variation_cnn(start, end, step, is_cnn):
    channel_count = list(range(start,end,step))
    result = []
    for h_channels in channel_count:
        print("parameter: ", h_channels)
        _, test_acc = eval_CNN(hidden_channels=h_channels, num_epochs=40, lr=1e-3, momentum=0.9)
        x = test_acc.detach().cpu().item()
        result.append(x)
    print(result)
    title = "Accuracy of CNN vs. # of hidden channels"
    xlabel = "# of hidden channels"
    ylabel = "Accuracy"
    display_result(channel_count, result, title, xlabel, ylabel)


# fnn_model, _ = eval_FNN(hidden_dim = 50, num_epochs = 40, lr = 1e-2, momentum = 0.9)
# cnn_model, _ = eval_CNN(hidden_channels=20, num_epochs=40, lr=1e-3, momentum=0.9)
# cnn_fnn_demo(fnn = fnn_model, cnn = cnn_model)
parameter_variation_cnn(2,22,2)