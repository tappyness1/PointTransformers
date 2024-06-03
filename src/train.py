import torch
from torch.utils.data import DataLoader, Subset
# from torchsummary import summary
from tqdm import tqdm
import numpy as np
from src.model.PTv1.model import PointTransformerClassifier
from src.data_processing.dataset import ModelNetDataset, ShapeNetDataset
from src.metrics import process_confusion_matrix
from src.validation import validation_classifier, validation_segmentation

import torch.nn as nn
import torch.optim as optim

def train_classifier(train_set, val_set, cfg, num_classes = 40):

    loss_function = nn.CrossEntropyLoss()
    
    network = PointTransformerClassifier(npoints = cfg['npoints'], n_classes = cfg['train']['num_classes'], in_dim = cfg['in_dim'])

    network.train()

    # optimizer = optim.Adam(network.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    # optimizer = optim.RMSprop(network.parameters(),
    #                         lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'], momentum=cfg['train']['weight_decay'], foreach=True)
    optimizer = optim.SGD(network.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'], momentum=cfg['train']['momentum'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)   

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    network = network.to(device)

    # if cfg['show_model_summary']:
    #     summary(network, (1024, 3))

    if cfg['train']['train_subset']:
        subset_indices = torch.randperm(len(train_set))[:cfg['train']['train_subset']]
        train_set = Subset(train_set, subset_indices)
    
    train_dataloader = DataLoader(train_set, batch_size=8, shuffle = True)

    for epoch in range(cfg['train']['epochs']):
        print (f"Epoch {epoch + 1}:")
        # for i in tqdm(range(X.shape[0])):
        with tqdm(train_dataloader) as tepoch:
            for point_clouds, labels in tepoch:
                # print (point_clouds.shape)
                # print (labels)

                optimizer.zero_grad() 
                out = network(point_clouds.to(device))
                loss = loss_function(out, labels.to(device))
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
        
        loss = validation_classifier(network, val_set, cfg, get_metrics = False)
        scheduler.step(loss)
        network.train()
        if epoch // cfg['train']['save_checkpoint_interval'] == 0:
            torch.save(network.state_dict(), f"{cfg['save_model_path']}_{epoch}.pt")
        
    print("training done")
    torch.save(network.state_dict(), f"{cfg['save_model_path']}_final.pt")

    print("Validating dataset")
    validation_classifier(network, val_set, cfg, get_metrics = True)

    return network

def get_final_results_classifier(val_cfg = None, cfg = None):
    """To be used if model weights are trained and just want to generate final results

    Args:
        val_cfg (_type_, optional): _description_. Defaults to None.
        cfg (_type_, optional): _description_. Defaults to None.
    """
    if not val_cfg:
        val_cfg = {"data_path": "../data/modelnet40_normal_resampled", "train": False, "modelnet_type": "modelnet10", "npoints": 1024}

    if not cfg:
        cfg = {"save_model_path": "model_weights/model_weights",
        'show_model_summary': True, 
        'npoints': 1024,
        'in_dim': 3, 
        'train': {"epochs": 10, 'lr': 1e-4, 
                    'weight_decay': 1e-4, 'momentum':0.9, 
                    'train_subset': 3990, # set 3990 for ModelNet10 else False if not intending to use subset. Set to 20 or something for small dataset experimentation/debugging
                    'val_subset': 906, # set 906 for ModelNet10, False otherwise
                    'num_classes': 10} # ModelNet40 so 40 classes, whereas ModelNet10 so 10 classes
        }
    
    val_set = ModelNetDataset(val_cfg)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_weights = torch.load(f"{cfg['save_model_path']}_final.pt", map_location=torch.device(device))
    model = PointTransformerClassifier(npoints = cfg['npoints'], n_classes = cfg['train']['num_classes'], in_dim = cfg['in_dim'])
    model.load_state_dict(model_weights)

    validation_classifier(model, val_set, cfg, get_metrics = True)


if __name__ == "__main__":

    torch.manual_seed(42)

    # # for local/VM runs
    # cfg = {"data_path": "../data/modelnet40_normal_resampled", "train": True, "modelnet_type": "modelnet10", "npoints": 1024}
    # train_set = ModelNetDataset(cfg)
    # cfg = {"data_path": "../data/modelnet40_normal_resampled", "train": False, "modelnet_type": "modelnet10", "npoints": 1024}
    # val_set = ModelNetDataset(cfg)

    # for Colab runs
    cfg = {"data_path": "/content/modelnet40_normal_resampled", "train": True, "modelnet_type": "modelnet10", "npoints": 1024}
    train_set = ModelNetDataset(cfg)
    cfg = {"data_path": "/content/modelnet40_normal_resampled", "train": False, "modelnet_type": "modelnet10", "npoints": 1024}
    val_set = ModelNetDataset(cfg)

    cfg = {"save_model_path": "model_weights/model_weights",
           'show_model_summary': True, 
           'npoints': 1024,
           'in_dim': 3, 
           'train': {"epochs": 10, 'lr': 1e-4, 
                     'weight_decay': 1e-4, 'momentum':0.9,
                     'save_checkpoint_interval': 5, 
                     'train_subset': 3990, # set 3990 for ModelNet10 else False if not intending to use subset. Set to 20 or something for small dataset experimentation/debugging
                     'val_subset': 906, # set 906 for ModelNet10, False otherwise
                     'num_classes': 10} # ModelNet40 so 40 classes, whereas ModelNet10 so 10 classes
            }
    train_classifier(train_set = train_set, val_set = val_set,  cfg = cfg, num_classes = cfg['train']['num_classes'])

    # # for Colab runs segmentations
    # cfg = {"data_path": "/content/shapenetcore_partanno_segmentation_benchmark_v0_normal", 
    #        "train": True, 
    #        "instance": "02691156",
    #        "shape_cut_off": 2500}
        
    # train_set = ShapeNetDataset(cfg)
    
    # cfg = {"data_path": "/content/shapenetcore_partanno_segmentation_benchmark_v0_normal", 
    #        "train": False, 
    #        "instance": "02691156",
    #        "shape_cut_off": 2500}    
    
    # val_set = ShapeNetDataset(cfg)

    # cfg = {"save_model_path": "model_weights/shapenet_airplane_model_weights.pt",
    #        'show_model_summary': True, 
    #        'train': {"epochs": 10, 'lr': 1e-4, 
    #                  'weight_decay': 1e-8, 'momentum':0.999, 
    #                  'train_subset': False, # set False if not intending to use subset. Set to 20 or something for small dataset experimentation/debugging
    #                  'val_subset': False, # see above
    #                  'num_classes': 4} # 4 due to only training on airplane
    #         }
    
    # train_segmentation(train_set = train_set, val_set = val_set, cfg = cfg, num_classes = cfg['train']['num_classes'])
