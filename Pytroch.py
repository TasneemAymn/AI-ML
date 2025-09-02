import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import torchvision.io as tv_io
import glob
from PIL import Image
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

from torchvision.models import vgg16
from torchvision.models import VGG16_Weights
weights = VGG16_Weights.DEFAULT
vgg_model = vgg16(weights=weights)

vgg_model.requires_grad_(False)
next(iter(vgg_model.parameters())).requires_grad
vgg_model.classifier[0:3]

N_CLASSES = 32

my_model = nn.Sequential(
    vgg_model.features,
    vgg_model.avgpool,
    nn.Flatten(),
    vgg_model.classifier[0:3],
    nn.Linear(4096, 500),
    nn.ReLU(),
    nn.Linear(500, N_CLASSES)
)

my_model

loss_function = nn.CrossEntropyLoss()
optimizer = Adam(my_model.parameters())
my_model = torch.compile(my_model.to(device))

pre_trans = weights.transforms()
IMG_WIDTH, IMG_HEIGHT = (224, 224)

random_trans = transforms.Compose([
    transforms.RandomHorizontalFlip()
])

DATA_LABELS = ["freshapples", "freshbanana", "freshoranges", 
               "rottenapples", "rottenbanana", "rottenoranges"]

class MyDataset(Dataset):
    def __init__(self, data_dir):
        self.imgs = []
        self.labels = []
        
        for l_idx, label in enumerate(DATA_LABELS):
            data_paths = glob.glob(data_dir + label + '/*.png', recursive=True)
            for path in data_paths:
                img = tv_io.read_image(path, tv_io.ImageReadMode.RGB)
                self.imgs.append(pre_trans(img).to(device))
                self.labels.append(torch.tensor(l_idx).to(device))

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.imgs)


n = 32

train_path = "data/fruits/train/"
train_data = MyDataset(train_path)
train_loader = DataLoader(train_data, batch_size=n, shuffle=True)
train_N = len(train_loader.dataset)

valid_path = "data/fruits/valid/"
valid_data = MyDataset(valid_path)
valid_loader = DataLoader(valid_data, batch_size=n)
valid_N = len(valid_loader.dataset)


epochs = 10
for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    utils.train(my_model, train_loader, train_N, random_trans, optimizer, loss_function)
    utils.validate(my_model, valid_loader, valid_N, loss_function)


vgg_model.requires_grad_(True)
optimizer = Adam(my_model.parameters(), lr=0.0001)

epochs = 1
for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    utils.train(my_model, train_loader, train_N, random_trans, optimizer, loss_function)
    utils.validate(my_model, valid_loader, valid_N, loss_function)


utils.validate(my_model, valid_loader, valid_N, loss_function)
