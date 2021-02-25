import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os 
import natsort
import cv2 
from PIL import Image


class Expand(object):
    """ Custom pytorch transform class to fit a 24x24 px writing image into a 28x28 px black canvas """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample.numpy()
        h, w = image.shape[1], image.shape[2]
        expand_s = self.output_size
        background = np.zeros((1, expand_s, expand_s))
        rnd_x = 1
        rnd_y = 1
        background[:,rnd_x:rnd_x+w,rnd_y:rnd_y+h] = image
        return torch.from_numpy(background)


class Net(nn.Module):
    """ Class of pytorch training model """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 25, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(25, 50, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(50*4*4, 384)
        self.fc2 = nn.Linear(384, 128)
        self.fc3 = nn.Linear(128, 27)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 50*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class UserImageDataSet(Dataset):
    """ Helper class to load image from a given path at `root_dit` """
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.all_imgs = os.listdir(root_dir)
        self.total_imgs = natsort.natsorted(self.all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        for filename in self.all_imgs:
            if filename.split('_')[0] == str(idx):
                path = os.path.join(self.root_dir, filename)
                image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                image = np.rot90(image, axes=(0,1))
                image = cv2.resize(image, (28,28))
                image = Image.fromarray(image)
                image = self.transform(image)
                return image


class Classifier():
    """ Class to classify a letter writen on a 24x24 image
    The background color should be black, and the text color should be white.
    The default pytorch model is `models/model_letters.pth`.
    The default image folder path is `user_images/`.
    """
    def __init__(self, path=None, img_folder_path=None):
        # upper letters A~Z; notice that the first member in this list is meaningless
        self.classes = [chr(i) for i in range(64,91)] 
        self.transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Resize(25),
                    Expand(28),
                    transforms.Normalize((0.5), (0.5))])

        # load pytorch model
        self.net = Net()
        self.path = 'models/model_letters.pth' if path is None else path
        self.net.load_state_dict(torch.load(self.path))

        self.img_folder_path = 'user_images' if img_folder_path is None else img_folder_path


    def classify(self, image_name, pred_idx=0):
        """ Predict letter on an image at given name
        The image_name should be in format `<idx_in_folder>_<random_string>.png`. Calling this function
        will classify on the image that have the same index numer as <idx_in_folder> when loading 
        files from `user_images/`.

            Args:
                image_name: name of saved image to classify 
                pred_idx: if pred_idx > 0, the predict letter will be the letter with
                         <pred_idx>th largest possibility from net
        """
        user_dataset = UserImageDataSet(self.img_folder_path, transform=self.transform)
        image = user_dataset[int(image_name.split('_')[0])]
        image = image.unsqueeze(0)
        output = self.net(image.float())
        if pred_idx == 0:
            _, predicted = torch.max(output, 1)
            return self.classes[predicted[0]]
        else:
            output = output.tolist()[0]
            for i in range(pred_idx):
                max_idx = output.index(max(output))
                output[max_idx] = min(output)-1
            predicted = output.index(max(output))
            return self.classes[predicted]


