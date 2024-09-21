import numpy as np
from model import Net
fron utils inport print_quant_measures
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms2
from torch.autograd import Variable

if __main__(path_to_feature1,path_to_featur2):
	feature1 = np.load(path_to_feature2)
	feature2 = np.load(path_to_feature1)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	Model1 = Net(n_class = 1,out_normalize = True)
	Model1 = Model1.to(device)




	feat1 = torch.from_numpy(feature1).to(dtype=torch.float)
	feat2 = torch.from_numpy(feature2).to(dtype=torch.float)	
	Iage,Igen,Ifeat1,Ifeat2,mean,log_var = Model1(feat1,feat2)

	print('Predicted Age : ',Iage)
	print('Predicted Gender : ',Igen)