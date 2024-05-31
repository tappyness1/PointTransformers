import torch.nn as nn
import torch
import pandas as pd

def predict(model, imgs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    preds = model(imgs)
    return torch.argmax(preds, dim = 1)
