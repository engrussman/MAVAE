import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.e_11 = nn.Linear(800, 500)
        self.e_12 = nn.Linear(500, 200)
        self.e_13 = nn.Linear(200, 120)
        self.e_14 = nn.Linear(120, 60)

        self.e_21 = nn.Linear(800, 500)
        self.e_22 = nn.Linear(500, 200)
        self.e_23 = nn.Linear(200, 120)
        self.e_24 = nn.Linear(120, 60)

        self.mean_layer = nn.Linear(120, 2)
        self.logvar_layer = nn.Linear(120, 2)

        self.d_11 = nn.Linear(60, 120)
        self.d_12 = nn.Linear(120, 200)
        self.d_13 = nn.Linear(200, 500)
        self.d_14 = nn.Linear(500, 800)


        self.d_21 = nn.Linear(60, 120)
        self.d_22 = nn.Linear(120, 200)
        self.d_23 = nn.Linear(200, 500)
        self.d_24 = nn.Linear(500, 800)


        self.c_11 = nn.Linear(120, 360)
        self.c_12 = nn.Linear(360, 150)
        self.c_13 = nn.Linear(150, 1)
        self.c_21 = nn.Linear(120, 120)
        self.c_22 = nn.Linear(120, 100)
        self.c_23 = nn.Linear(100, 1)

    def forward(self, f1,f2):
        x1 = F.relu(self.e_11(f1))
        x1 = F.relu(self.e_12(x1))
        x1 = F.relu(self.e_13(x1))
        x1 = F.relu(self.e_14(x1))


        x2 = F.relu(self.e_21(f2))
        x2 = F.relu(self.e_22(x2))
        x2 = F.relu(self.e_23(x2))
        x2 = F.relu(self.e_24(x2))


        x3 = F.relu(self.d_11(x1))
        x3 = F.relu(self.d_12(x3))
        x3 = F.relu(self.d_13(x3))
        x3 = F.relu(self.d_14(x3))



        x4 = F.relu(self.e_11(x2))
        x4 = F.relu(self.e_12(x4))
        x4 = F.relu(self.e_13(x4))
        x4 = F.relu(self.e_14(x4))


        xE = torch.cat((x1, x2), 1)
        mean = self.mean_layer(xE)
        var = self.logvar_layer(xE)
        

        xC1 = F.relu(self.c_11(xE))
        xC1 = F.relu(self.c_12(xC1))
        xC1 = F.relu(self.c_13(xC1))

        xC2 = F.relu(self.c_21(xE))
        xC2 = F.relu(self.c_22(xC2))
        xC2 = F.relu(self.c_23(xC2))

        return xC1,xC2,x3,x4,mean,var