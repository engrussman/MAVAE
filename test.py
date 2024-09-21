import numpy as np
from model import Net
fron utils inport print_quant_measures
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms2
from torch.autograd import Variable


features1 = np.load('Test_features1.npy')
features2 = np.load('Test_features2.npy')
labels_Age = np.load('Test_labels_Age.npy')
labels_Gender = np.load('Test_labels_Age.npy')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Model1 = Net(n_class = 1,out_normalize = True)
Model1 = Model1.to(device)


Predicted_age = []
Inferenced_age = []
Predicted_Gender = []
Inferenced_Gender = []

for index in range(len(feature1)):
	feature1 = features1[index]
	feature2 = features2[index]
	label_Age = labels_Age[index]
	label_Gender = labels_Gender[index]

	feat1 = torch.from_numpy(feature1).to(dtype=torch.float)
	geat22 = torch.from_numpy(feature2).to(dtype=torch.float)
	lab1 = torch.from_numpy(label_Age).to(dtype=torch.float)
	lab2 = torch.from_numpy(label_Gender).to(dtype=torch.float)
	
        Iage,Igen,Ifeat1,Ifeat2,mean,log_var = Model1(feat1,feat2)

	Predicted_age.append(Iage)
	Inferenced_age.append(Igen)
	Predicted_Gender.append(lab1)
	Inferenced_Gender.append(lab2)
	

print_quant_measures(Predicted_age,Inferenced_age,Predicted_Gender,Inferenced_Gender)
