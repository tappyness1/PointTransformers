import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from src.metrics import process_confusion_matrix
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import numpy as np

def get_accuracy(preds, ground_truth):
    ground_truth = ground_truth.squeeze(dim=1)
    preds = preds.argmax(dim=1)
    
    return (preds.flatten()==ground_truth.flatten()).float().mean()

def validation_classifier(model, val_set, cfg, get_metrics = False):
    """Simple validation workflow. Current implementation is for F1 score

    Args:
        model (_type_): _description_
        val_set (_type_): _description_

    Returns:
        _type_: _description_
    """
    model.eval()

    if cfg['train']['val_subset']:
        subset_indices = torch.randperm(len(val_set))[:cfg['train']['val_subset']]
        val_set = Subset(val_set, subset_indices)

    val_dataloader = DataLoader(val_set, batch_size=5, shuffle = True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    preds, gt = [],[]
    losses = []
    loss_function = nn.CrossEntropyLoss()

    with tqdm(val_dataloader) as tepoch:

        for point_clouds, labels in tepoch:
            
            with torch.no_grad():
                out = model(point_clouds.to(device))
            loss = loss_function(out, labels.to(device)) 
            tepoch.set_postfix(loss=loss.item())  
            losses.append(loss.item())
            if get_metrics:
                preds.append(torch.argmax(out, dim = 1).flatten())
                gt.append(labels)

    if get_metrics:
        # print (torch.cat(preds))
        preds = torch.cat(preds).cpu()
        gt = torch.cat(gt).cpu()
        cm = process_confusion_matrix(preds, gt, num_classes = cfg['train']['num_classes'])
        cm = pd.DataFrame(cm)
        print (f"Confusion Matrix: \n{cm}")
        cm.to_csv("val_results/classifier_confusion_matrix.csv", header=False, index=False)

        cr = classification_report(gt, preds, labels = np.arange(0,cfg['train']['num_classes'],1), output_dict = True)
        cr_print = classification_report(gt, preds, labels = np.arange(0,cfg['train']['num_classes'],1), output_dict = False)
        print (f"Classification Report: \n{cr_print}")
        cr = pd.DataFrame(cr).transpose()
        cr.to_csv("val_results/classifier_classification_report.csv", header=False, index=False)

    print (f"Validation Loss: {sum(losses)/len(losses)}")

    return sum(losses)/len(losses)

def validation_segmentation(model, val_set, cfg, get_metrics = False):
    """Simple validation workflow. Current implementation is for F1 score

    Args:
        model (_type_): _description_
        val_set (_type_): _description_

    Returns:
        _type_: _description_
    """
    model.eval()

    if cfg['train']['val_subset']:
        subset_indices = torch.randperm(len(val_set))[:cfg['train']['val_subset']]
        val_set = Subset(val_set, subset_indices)

    val_dataloader = DataLoader(val_set, batch_size=5, shuffle = True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    preds, gt = [],[]
    losses = []
    loss_function = nn.CrossEntropyLoss()

    with tqdm(val_dataloader) as tepoch:

        for point_clouds, labels in tepoch:
            
            with torch.no_grad():
                out_pos, out = model(point_clouds.to(device))
            loss = loss_function(out.permute(0,2,1), labels.to(torch.long).to(device)) 
            tepoch.set_postfix(loss=loss.item())  
            losses.append(loss.item())
            if get_metrics:
                preds.append(torch.argmax(out, dim = 2).flatten())
                gt.append(labels.flatten())

    if get_metrics:
        # print (torch.cat(preds))
        preds = torch.cat(preds).cpu()
        gt = torch.cat(gt).cpu()
        print (preds.shape)
        print (gt.shape)
        cm = process_confusion_matrix(preds, gt, num_classes = cfg['train']['num_classes'])
        cm = pd.DataFrame(cm)
        print (f"Confusion Matrix: \n{cm}")
        cm.to_csv("val_results/segmentation_confusion_matrix.csv", header=False, index=False)

        cr = classification_report(gt, preds, labels = np.arange(0,cfg['train']['num_classes'],1), output_dict = True)
        print (f"Classification Report: \n{cr}")
        cr = pd.DataFrame(cr).transpose()
        cr.to_csv("val_results/segmentation_classification_report.csv", header=False, index=True)

    print (f"Validation Loss: {sum(losses)/len(losses)}")

    return sum(losses)/len(losses)


if __name__ == "__main__":
    
    from src.dataset import get_load_data

    _, val_set = get_load_data(root = "../data", dataset = "VOCSegmentation")
    trained_model_path = "model_weights/model_weights.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(trained_model_path, map_location=torch.device(device))

    cfg = {"save_model_path": "model_weights/model_weights.pt",
           'show_model_summary': True, 
           'train': {"epochs": 10, 'lr': 1e-3, 
                     'weight_decay': 1e-8, 'momentum':0.999, 
                     'subset': False, # set False if not intending to use subset. Set to 20 or something for small dataset experimentation/debugging
                     'num_classes': 40} # ModelNet40 so 40 classes
            }
    
    validation_classifier(model, val_set, cfg)