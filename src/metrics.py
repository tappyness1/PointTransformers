import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def process_confusion_matrix(out: torch.Tensor, gt: torch.Tensor, num_classes):
    """
    This function computes the confusion matrix for a given output and ground truth.
    
    Args:
    out: torch.Tensor: The output of the model
    gt: torch.Tensor: The ground truth
    
    Returns:
    torch.Tensor: The confusion matrix
    """
    return confusion_matrix(out, gt, labels= np.arange(0,num_classes,1))


if __name__ == "__main__":
    out = torch.argmax(torch.rand(5, 40, 1), dim = 1).flatten()
    gt = torch.randint(0, 40, (5,1)).flatten() # don't need to flatten in reality
    print (process_confusion_matrix(out, gt, 40))