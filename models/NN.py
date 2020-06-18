# -*- coding: utf-8 -*-
# pytorch 版本的
import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F

class RegressionNN(nn.Module):
    # A simple MLP regression network 
    def __init__(self, feature_number):
        super(RegressionNN,self).__init__()
        self.feature_number = feature_number
        self.fc1 = nn.Linear(self.feature_number,12)
        self.fc2 = nn.Linear(12,8)
        self.fc3 = nn.Linear(8,1)
        
    def forward(self,x):
        x = self.fc1(x)
        # x = F.sigmoid(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        return x
             
    def predict(self,X_test):
        # 这里要做成尽量贴近sklearn的模式
        X_test = np.array(X_test)
        # 然后 to tensor 
        X_test  = torch.from_numpy(X_test).type(torch.FloatTensor)
        # pred = F.softmax(self.forward(x), dim= 1) # 可能是这里不对
        y_pred = self.forward(X_test) # now torch tensor 
        y_pred = y_pred.detach().numpy()
        return y_pred.flatten() # 这里直接拉成1d的vector
