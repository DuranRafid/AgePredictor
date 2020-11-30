# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
import umap
from Regularizers import ElasticNetRegularizer

# Create Fully Connected Network
class Linear_Regression_NN(nn.Module):
    def __init__(self, in_size, out_size):
        super(Linear_Regression_NN, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.fc1 = nn.Linear(self.in_size, 3) # self.out_size)
        self.fc2 = nn.Linear(3,self.out_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class Neural_Regression:
    def __init__(self, in_size=5, out_size=1, learning_rate=0.001, num_epochs=100, reg='ElasticNet'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Linear_Regression_NN(in_size, out_size).to(device=self.device)
        self.lr = learning_rate
        self.num_epochs = num_epochs

    def fit(self, data, targets):
        data = np.array(data, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        # Preprocess
        data = torch.tensor(data, dtype=torch.float32, device=self.device)
        targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
        targets = targets.reshape(targets.shape[0],1)
        criterion = nn.MSELoss()
        optmizer = optim.SGD(self.model.parameters(), lr=self.lr)
        regularizer_loss = ElasticNetRegularizer(model=self.model,alpha_reg=1.0, lambda_reg=0)

        for epoch in range(self.num_epochs):
            # Forward
            scores = self.model(data)
            loss = criterion(scores, targets)
            loss = regularizer_loss.regularized_all_param(reg_loss_function=loss)
            # Backward
            optmizer.zero_grad()
            loss.backward()

            # Gradient Descent or Adam step
            optmizer.step()
        return self

    def predict(self, data):
        data = np.array(data, dtype=np.float32)
        data = torch.tensor(data, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(data).cpu()

        return pred.detach().numpy()
