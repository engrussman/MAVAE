import numpy as np
from model import Net
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms2
from torch.autograd import Variable


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        self.files = root

    def __getitem__(self, index):
	global features1
	global features2
	global labels_Age
	global labels_Gender
	feature1 = features1[index]
	feature2 = features2[index]
	label_Age = labels_Age[index]
	label_Gender = labels_Gender[index]

        feature1 = torch.from_numpy(feature1).to(dtype=torch.float)
        feature2 = torch.from_numpy(feature2).to(dtype=torch.float)
        label_Age = torch.from_numpy(label_Age).to(dtype=torch.float)
        label_Gender = torch.from_numpy(label_Gender).to(dtype=torch.float)
        return {"feature1": feature1,"feature2": feature2, "label_Age": label_Age, "label_Gender": label_Gender}

    def __len__(self):
        return len(self.files)


features1 = np.load('features1.npy')
features2 = np.load('features2.npy')
labels_Age = np.load('labels_Age.npy')
labels_Gender = np.load('labels_Age.npy')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Model1 = Net(n_class = 1,out_normalize = True)
Model1 = Model1.to(device)

optimizer_1 = torch.optim.Adam(Model1.parameters(), lr=0.0002, betas=(0.9, 0.99))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

train_dataloader = DataLoader(
    ImageDataset(list(range(15000)), hr_shape=hr_shape),
    batch_size=b_size,
    shuffle=True)

val_dataloader = DataLoader(
    ImageDataset(list(range(15000)), hr_shape=hr_shape),
    batch_size=b_size,
    shuffle=True)

total_epoch = 400
best_val_loss = 123.69739469567173

criterion_content1 = torch.nn.L1Loss().to(device)
criterion_content2 = torch.nn.L1Loss().to(device)
criterion_content3 = torch.nn.L1Loss().to(device)
criterion_content4 = torch.nn.L1Loss().to(device)

for epoch in range(total_epoch):
    train_loss = 0
    Model1.train()  # Set the model to training mode
    for i, imgs in tqdm(enumerate(train_dataloader)):
        feat1 = Variable(imgs["feature1"].type(Tensor))
        feat2 = Variable(imgs["feature2"].type(Tensor))
        lab1 = Variable(imgs["label_Age"].type(Tensor))
	lab2 = Variable(imgs["label_Gender"].type(Tensor))
        optimizer_1.zero_grad()
        Iage,Igen,Ifeat1,Ifeat2,mean,log_var = Model1(feat1,feat2)

        # Measure pixel-wise loss against ground truth
        loss1 = criterion_content1(Ifeat1, feat1)
        loss2 = criterion_content1(Ifeat2, feat2)
        loss3 = criterion_content1(Iage, lab1)
        loss4 = criterion_content1(Igen, lab2)
	KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        loss = loss1+loss2+loss3+loss4+KLD
        loss.backward()
        optimizer_1.step()