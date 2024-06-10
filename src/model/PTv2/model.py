import torch
import torch.nn as nn

class PTV2Classifier(nn.Module):

    def __init__(self, npoints = 1000, n_classes = 10, in_dim = 6):
        super().__init__()

    def forward(self, points):
        pass

class PTV2Segmentation(nn.Module):
    
    def __init__(self, npoints = 1000):
        super().__init__()

    def forward(self, points):
        pass
    