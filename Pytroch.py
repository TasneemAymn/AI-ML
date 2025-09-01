import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import torchvision.io as tv_io
import glob
from PIL import Image
import utils

# Select device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()   # Check if CUDA (GPU) is available

# Import pretrained VGG16 model and its default weights (trained on ImageNet)
from torchvision.models import vgg16
from torchvision.models import VGG16_Weights
weights = VGG16_Weights.DEFAULT
vgg_model = vgg16(weights=weights)

# Freeze the entire VGG16 model so its pretrained weights won't update during training
vgg_model.requires_grad_(False)

# Check if the first parameter in the model is frozen (requires_grad == False)
next(iter(vgg_model.parameters())).requires_grad

# Inspect the first 3 layers of the classifier part in VGG16
vgg_model.classifier[0:3]

# Number of output classes in your dataset
N_CLASSES = 32

# Build a custom model by stacking pretrained VGG16 as a feature extractor
my_model = nn.Sequential(
    vgg_model.features,        # Convolutional feature extractor
    vgg_model.avgpool,         # Adaptive average pooling
    nn.Flatten(),              # Flatten features into a vector
    vgg_model.classifier[0:3], # First part of VGG16 classifier
    nn.Linear(4096, 500),      # New fully connected layer (reduce to 500 features)
    nn.ReLU(),                 # Activation function
    nn.Linear(500, N_CLASSES)  # Final layer to classify into 32 classes
)

my_model  # Show the architecture

# Define loss function (CrossEntropy for classification)
loss_function = nn.CrossEntropyLoss()

# Define optimizer (Adam will update weights of unfrozen layers only)
optimizer = Adam(my_model.parameters())

# Compile and move model to device (GPU if available)
my_model = torch.compile(my_model.to(device))

# Define preprocessing transforms from pretrained weights (resize, normalize, etc.)
pre_trans = weights.transforms()

# Standard image size for VGG16
IMG_WIDTH, IMG_HEIGHT = (224, 224)

# Define random data augmentation (flip images horizontally with 50% chance)
random_trans = transforms.Compose([
    transforms.RandomHorizontalFlip()
])
# Define dataset labels (classes)
# Each label corresponds to a folder name containing images
DATA_LABELS = ["freshapples", "freshbanana", "freshoranges", 
               "rottenapples", "rottenbanana", "rottenoranges"]

# Custom Dataset class for loading fruit images
class MyDataset(Dataset):
    def __init__(self, data_dir):
        self.imgs = []   # list to store all images
        self.labels = [] # list to store all labels
        
        # Loop through each label and its index
        for l_idx, label in enumerate(DATA_LABELS):
            # Get all image paths from subfolders
            data_paths = glob.glob(data_dir + label + '/*.png', recursive=True)
            for path in data_paths:
                # Read image as tensor (RGB format)
                img = tv_io.read_image(path, tv_io.ImageReadMode.RGB)
                # Apply pretrained VGG16 preprocessing (resize + normalize) and move to device
                self.imgs.append(pre_trans(img).to(device))
                # Save the label as tensor (integer)
                self.labels.append(torch.tensor(l_idx).to(device))

    # Return one sample (image + label) by index
    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        return img, label

    # Return dataset size
    def __len__(self):
        return len(self.imgs)


# ------------------ DATA LOADING ------------------ #
n = 32  # batch size

# Training dataset and loader
train_path = "data/fruits/train/"
train_data = MyDataset(train_path)
train_loader = DataLoader(train_data, batch_size=n, shuffle=True)
train_N = len(train_loader.dataset)  # total number of training samples

# Validation dataset and loader
valid_path = "data/fruits/valid/"
valid_data = MyDataset(valid_path)
valid_loader = DataLoader(valid_data, batch_size=n)
valid_N = len(valid_loader.dataset)  # total number of validation samples


# ------------------ FIRST TRAINING PHASE ------------------ #
epochs = 10
for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    # Train model on training data
    utils.train(my_model, train_loader, train_N, random_trans, optimizer, loss_function)
    # Validate model on validation data
    utils.validate(my_model, valid_loader, valid_N, loss_function)


# ------------------ FINE-TUNING PHASE ------------------ #
# Unfreeze the base VGG16 model so pretrained weights can be updated (fine-tuning)
vgg_model.requires_grad_(True)
# Re-define optimizer with a smaller learning rate for fine-tuning
optimizer = Adam(my_model.parameters(), lr=0.0001)

# Train again for a few epochs (here only 1 epoch for demonstration)
epochs = 1
for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    utils.train(my_model, train_loader, train_N, random_trans, optimizer, loss_function)
    utils.validate(my_model, valid_loader, valid_N, loss_function)


# ------------------ FINAL EVALUATION ------------------ #
# Run final validation after training
utils.validate(my_model, valid_loader, valid_N, loss_function)

# External assessment function (provided in run_assessment.py)
from run_assessment import run_assessment
run_assessment(my_model)

