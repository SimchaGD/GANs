import torch
import pandas as pd
import numpy as np
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.model_selection import train_test_split

class DataMalaria(Dataset):
    """
    filepath = directory naar afbeeldingen
    transform = eventuele transformatie naar pytorch tensors
    TTS = train test split. Het splitsen tussen trainingset en testset
    """
    def __init__(self, filepath, transform = None, TTS = True):
        self.data = pd.read_csv(filepath, sep = ";")
        self.transform = transform
        self.train = self.data
        self.test = self.data
        
        if TTS:
            self.trainTestSplit()
            self.trainmode()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        imname = self.data.iloc[index, 0]
        image = plt.imread("RealImg/{}".format(imname))
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image.double()
    
    def imshowsingle(self, index, ax = None):
        image= self[index]
        if ax == None:
            plt.imshow(torch.transpose(image, -3, 2))
            plt.show()
        else:
            ax.imshow(torch.transpose(image, -3, 2))
            
    
    
    def trainTestSplit(self, p = 0.25):
        self.train, self.test = train_test_split(self.data, test_size = p, random_state = 1)
        
    def testmode(self):
        self.data = self.test
        
    def trainmode(self):
        self.data = self.train
    