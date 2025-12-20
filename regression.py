import numpy as np
import matplotlib.pyplot as plt

import torch
import scipy.stats as sts
import learners

def gen_nonlinear_data(num_samples=100000):
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
    
def mse_loss(y_pred, y_true):
    square_diff = torch.pow((y_pred-y_true), 2)
    mean_error = 0.5 * torch.mean(square_diff)
    return mean_error

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
    optimizer = learners.CustomSGD(model.parameters(), lr=lr, momentum=momentum)  # create an Adam optimizer for the model parameters
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()  
        pred = model(xTr)  # run the forward pass through the model to compute predictions
        loss = mse_loss(pred, yTr)
        loss.backward()  # compute the gradient wrt loss
        optimizer.step()  # performs a step of gradient descent
        if display_loss and (epoch + 1) % print_freq == 0:
            print('epoch {} loss {}'.format(epoch+1, loss.item()))
    
    return model  # return trained model

def reg_lin_eval():
    ndims = nl_xTr.shape[1]
    linear_model = learners.LinearRegressionModel(ndims)  # initialize the model
    linear_model = train_regression_model(nl_xTr, nl_yTr,linear_model, num_epochs=2000, lr=1e-2, momentum=0.9, print_freq=500)
    avg_test_error_linreg = mse_loss(linear_model(nl_xTe), nl_yTe)  # compute the average test error
    print('linear regression avg test error', avg_test_error_linreg.item())
    return linear_model, avg_test_error_linreg.item()

def reg_fnn_eval(hdims, num_epochs, lr, momentum):
    hdims = 90
    num_epochs = 5000
    lr = 1e-1
    momentum = 0.9

    mlp_model = learners.MLPNet(input_dim=1, hidden_dim=hdims, output_dim=1)
    mlp_model = train_regression_model(nl_xTr, nl_yTr, mlp_model, num_epochs=num_epochs, lr=lr, momentum=momentum)
    avg_test_error_mlp = mse_loss(mlp_model(nl_xTe), nl_yTe)
    print('Feedforward neural network avg test error', avg_test_error_mlp.item())
    return mlp_model, avg_test_error_mlp

def reg_comparison():
    """
    Shows the sinusoidal regression by FNN and linear model 
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3))

    # Plot the visualizations from our MLP Model
    ax1.scatter(nl_xTr, nl_yTr, label="Train Points")
    ax1.scatter(nl_xTe, nl_yTe, label="Test Points")
    ax1.scatter(nl_xTr, mlp_model(nl_xTr).detach(), color="red", marker='o', label="Prediction")
    ax1.legend()
    ax1.set_title('FNN')

    # Plot the visualizations from our MLP Model
    ax2.scatter(nl_xTr, nl_yTr, label="Train Points")
    ax2.scatter(nl_xTe, nl_yTe, label="Test Points")
    ax2.plot(nl_xTr, linear_model(nl_xTr).detach(),linewidth=5.0, color="red", label="Prediction Line")
    ax2.legend()
    ax2.set_title('Linear Model')

    plt.show()

def sq_error(y_pred,y_true):
    return torch.pow((y_pred-y_true),2)

def t_test(linear_model, mlp_model):
    sq_err_lin = sq_error(linear_model(nl_xTe), nl_yTe).detach().numpy()
    sq_err_fnn = sq_error(mlp_model(nl_xTe), nl_yTe).detach().numpy()
    std_lin = np.std(sq_err_lin)
    std_fnn = np.std(sq_err_fnn)
    return sts.ttest_ind(sq_err_lin/std_lin, sq_err_fnn/std_fnn)

nl_xTr, nl_xTe, nl_yTr, nl_yTe = gen_nonlinear_data(num_samples=500)
linear_model, _ = reg_lin_eval()
mlp_model, _ = reg_fnn_eval(hdims = 90, num_epochs=5000, lr=1e-1, momentum = 0.9)
# result = t_test(linear_model, mlp_model)
# print(result)
reg_comparison()



