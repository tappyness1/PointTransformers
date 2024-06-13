import torch
from torch.utils.data import DataLoader, Subset
# from torchsummary import summary
from tqdm import tqdm
import numpy as np
from src.model.PTv1.model import PointTransformerClassifier, PointTransformerSegmentation
from src.model.PTv2.model import PTV2Classifier, PTV2Segmentation
from src.data_processing.dataset import ModelNetDataset, ShapeNetDataset
from src.metrics import process_confusion_matrix
from src.validation import validation_classifier, validation_segmentation

import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self):
        self.model = None
        self.train_set = None
        self.val_set = None
        self.cfg = None
        self.type = None

    def train_classifier(self):
        assert self.type == "Classifier", "This method is only for training classifiers"

        loss_function = nn.CrossEntropyLoss()
        
        self.model.train()

        # optimizer = optim.Adam(network.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
        # optimizer = optim.RMSprop(network.parameters(),
        #                         lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'], momentum=cfg['train']['weight_decay'], foreach=True)
        optimizer = optim.SGD(self.model.parameters(), lr=self.cfg['train']['lr'], weight_decay=self.cfg['train']['weight_decay'], momentum=self.cfg['train']['momentum'])

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)   

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = self.model.to(device)

        if self.cfg['train']['train_subset']:
            subset_indices = torch.randperm(len(self.train_set))[:self.cfg['train']['train_subset']]
            self.train_set = Subset(self.train_set, subset_indices)
        
        num_epochs = cfg['train']['epochs']

        train_dataloader = DataLoader(self.train_set , batch_size=8, shuffle = True)
        
        if cfg['train']['continue_training']:
            model_weights = torch.load(cfg['train']['weights_path'], map_location=torch.device(device))
            self.model.load_state_dict(model_weights)
            
            # get the saved epochs and continue training from there
            last_epoch = int(cfg['train']['weights_path'].split("_")[-1].split(".")[0])
            num_epochs -= last_epoch

        for epoch in range(num_epochs):
            print (f"Epoch {epoch + 1}:")
            # for i in tqdm(range(X.shape[0])):
            with tqdm(train_dataloader) as tepoch:
                for point_clouds, labels in tepoch:
                    # print (point_clouds.shape)
                    # print (labels)

                    optimizer.zero_grad() 
                    out = self.model(point_clouds.to(device))
                    loss = loss_function(out, labels.to(device))
                    loss.backward()
                    optimizer.step()
                    tepoch.set_postfix(loss=loss.item())
            
            loss = validation_classifier(self.model, self.val_set, cfg, get_metrics = False)
            scheduler.step(loss)
            self.model.train()
            if epoch % self.cfg['train']['save_checkpoint_interval'] == 0:
                torch.save(self.model.state_dict(), f"{self.cfg['save_model_path']}_classifier_{epoch}.pt")
            
        print("training done")
        torch.save(self.model.state_dict(), f"{self.cfg['save_model_path']}_classifier_final.pt")

        print("Validating dataset")
        validation_classifier(self.model, self.val_set, cfg, get_metrics = True)

        return self.model
    
    def get_final_results_classifier(self, val_cfg = None, cfg = None):
        """To be used if model weights are trained and just want to generate final results

        Args:
            val_cfg (_type_, optional): _description_. Defaults to None.
            cfg (_type_, optional): _description_. Defaults to None.
        """

        assert self.type == "Classifier", "This method is only for training classifiers"

        if not val_cfg:
            val_cfg = {"data_path": "../data/modelnet40_normal_resampled", "train": False, "modelnet_type": "modelnet10", "npoints": 1024}

        if not cfg:
            cfg = {"save_model_path": "model_weights/model_weights",
            'show_model_summary': True, 
            'npoints': 1024,
            'in_dim': 6, 
            'train': {"epochs": 10, 'lr': 1e-4, 
                        'weight_decay': 1e-4, 'momentum':0.9, 
                        'train_subset': 3990, # set 3990 for ModelNet10 else False if not intending to use subset. Set to 20 or something for small dataset experimentation/debugging
                        'val_subset': 906, # set 906 for ModelNet10, False otherwise
                        'num_classes': 10} # ModelNet40 so 40 classes, whereas ModelNet10 so 10 classes
            }
        
        val_set = ModelNetDataset(val_cfg)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_weights = torch.load(f"{cfg['save_model_path']}_classifier_final.pt", map_location=torch.device(device))
        self.model.load_state_dict(model_weights)

        validation_classifier(self.model, val_set, cfg, get_metrics = True)

    def train_segmentation(self):

        assert self.type == "Segmentation", "This method is only for training Segmentation Models"

        loss_function = nn.CrossEntropyLoss()
        self.model.train()

        # optimizer = optim.Adam(network.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
        # optimizer = optim.RMSprop(network.parameters(),
        #                         lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'], momentum=cfg['train']['weight_decay'], foreach=True)
        optimizer = optim.SGD(self.model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'], momentum=cfg['train']['momentum'])

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)   

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = self.model.to(device)

        # if cfg['show_model_summary']:
        #     summary(network, (1024, 3))

        if cfg['train']['train_subset']:
            subset_indices = torch.randperm(len(self.train_set))[:self.cfg['train']['train_subset']]
            self.train_set = Subset(self.train_set, subset_indices)
        
        train_dataloader = DataLoader(self.train_set, batch_size=8, shuffle = True)

        num_epochs = cfg['train']['epochs']
        
        if cfg['train']['continue_training']:
            model_weights = torch.load(cfg['train']['weights_path'], map_location=torch.device(device))
            self.model.load_state_dict(model_weights)
            
            # get the saved epochs and continue training from there
            last_epoch = int(cfg['train']['weights_path'].split("_")[-1].split(".")[0])
            num_epochs -= last_epoch

        for epoch in range(num_epochs):
            print (f"Epoch {epoch + 1}:")
            # for i in tqdm(range(X.shape[0])):
            with tqdm(train_dataloader) as tepoch:
                for point_clouds, labels in tepoch:
                    # print (point_clouds.shape)
                    # print (labels)

                    optimizer.zero_grad() 
                    out_pos, out = self.model(point_clouds.to(device))
                    loss = loss_function(out.permute(0,2,1), labels.to(torch.long).to(device))
                    loss.backward()
                    optimizer.step()
                    tepoch.set_postfix(loss=loss.item())
            
            loss = validation_segmentation(self.model, self.val_set, cfg, get_metrics = False)
            scheduler.step(loss)
            self.model.train()
            if epoch % cfg['train']['save_checkpoint_interval'] == 0:
                torch.save(self.model.state_dict(), f"{self.cfg['save_model_path']}_segmentation_{epoch}.pt")
            
        print("training done")
        torch.save(self.model.state_dict(), f"{self.cfg['save_model_path']}_segmentation_final.pt")

        print("Validating dataset")
        validation_segmentation(self.model, self.val_set, cfg, get_metrics = True)

        return self.model

    def get_final_results_segmentation(self, val_cfg = None, cfg = None):
        """To be used if model weights are trained and just want to generate final results

        Args:
            val_cfg (_type_, optional): _description_. Defaults to None.
            cfg (_type_, optional): _description_. Defaults to None.
        """
        if not val_cfg:
            val_cfg = {"data_path": "/content/shapenetcore_partanno_segmentation_benchmark_v0_normal", 
            "train": False, 
            "instance": "02691156",
            "shape_cut_off": 2500}    

        if not cfg:
            cfg = {"save_model_path": "model_weights/shapenet_airplane_model_weights",
                    'show_model_summary': True, 
                    'npoints': 2500,
                    'in_dim': 6,
                    'train': {"epochs": 10, 'lr': 0.05, 
                                'weight_decay': 1e-8, 'momentum':0.999, 
                                'save_checkpoint_interval': 5,
                                'train_subset': False, # set False if not intending to use subset. Set to 20 or something for small dataset experimentation/debugging
                                'val_subset': False, # see above
                                'num_classes': 4} # 4 due to only training on airplane
                        }
        
        val_set = ShapeNetDataset(val_cfg)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_weights = torch.load(f"{cfg['save_model_path']}_segmentation_final.pt", map_location=torch.device(device))
        self.model.load_state_dict(model_weights)

        validation_segmentation(self.model, val_set, cfg, get_metrics = True)
    
class PTClassifierTrainer(Trainer):
    def __init__(self, train_set, val_set, cfg):
        self.model = PointTransformerClassifier(npoints = cfg['npoints'], n_classes = cfg['train']['num_classes'], in_dim = cfg['in_dim'])
        self.train_set = train_set
        self.val_set = val_set
        self.cfg = cfg
        self.type = "Classifier"

class PTV2ClassifierTrainer(Trainer):
    def __init__(self, train_set, val_set, cfg):
        self.model = PTV2Classifier(n_classes = cfg['train']['num_classes'], in_dim = cfg['in_dim'])
        self.train_set = train_set
        self.val_set = val_set
        self.cfg = cfg
        self.type = "Classifier"

class PTSegmentationTrainer(Trainer):
    def __init__(self, train_set, val_set, cfg):
        self.model = PointTransformerSegmentation(npoints = cfg['npoints'], n_classes = cfg['train']['num_classes'], in_dim = cfg['in_dim'])
        self.train_set = train_set
        self.val_set = val_set
        self.cfg = cfg
        self.type = "Segmentation"

class PTV2SegmentationTrainer(Trainer):
    def __init__(self, train_set, val_set, cfg):
        self.model = PTV2Segmentation(n_classes = cfg['train']['num_classes'], in_dim = cfg['in_dim'])
        self.train_set = train_set
        self.val_set = val_set
        self.cfg = cfg
        self.type = "Segmentation"

if __name__ == "__main__":

    torch.manual_seed(42)

    # # for local/VM runs
    # cfg = {"data_path": "../data/modelnet40_normal_resampled", "train": True, "modelnet_type": "modelnet10", "npoints": 1024}
    # train_set = ModelNetDataset(cfg)
    # cfg = {"data_path": "../data/modelnet40_normal_resampled", "train": False, "modelnet_type": "modelnet10", "npoints": 1024}
    # val_set = ModelNetDataset(cfg)

    # # for Colab runs
    # cfg = {"data_path": "/content/modelnet10_normal_resampled", "train": True, "modelnet_type": "modelnet10", "npoints": 1024}
    # train_set = ModelNetDataset(cfg)
    # cfg = {"data_path": "/content/modelnet10_normal_resampled", "train": False, "modelnet_type": "modelnet10", "npoints": 1024}
    # val_set = ModelNetDataset(cfg)

    # cfg = {"save_model_path": "model_weights/model_weights",
    #        'show_model_summary': True, 
    #        'npoints': 1024,
    #        'in_dim': 6, 
    #        'train': {"epochs": 10, 'lr': 1e-4, 
    #                  'weight_decay': 1e-4, 'momentum':0.9,
    #                  'save_checkpoint_interval': 5, 
    #                  'train_subset': 3990, # set 3990 for ModelNet10 else False if not intending to use subset. Set to 20 or something for small dataset experimentation/debugging
    #                  'val_subset': 906, # set 906 for ModelNet10, False otherwise
    #                  'num_classes': 10, # ModelNet40 so 40 classes, whereas ModelNet10 so 10 classes
    #                  'continue_training': True,
    #                  'weights_path': "model_weights/model_weights_classifier_4.pt"} 
    #         }

    # # classifier_trainer = PTClassifierTrainer(train_set = train_set, val_set = val_set, cfg = cfg)
    # # PTClassifierTrainer().train_classifier()
    # classifier_trainer = PTV2ClassifierTrainer(train_set = train_set, val_set = val_set, cfg = cfg)
    # classifier_trainer.train_classifier()

    # for Colab runs segmentations
    cfg = {"data_path": "/content/shapenetcore_partanno_segmentation_benchmark_v0_normal", 
           "train": True, 
           "instance": "02691156",
           "shape_cut_off": 2500}
        
    train_set = ShapeNetDataset(cfg)
    
    cfg = {"data_path": "/content/shapenetcore_partanno_segmentation_benchmark_v0_normal", 
           "train": False, 
           "instance": "02691156",
           "shape_cut_off": 2500}    
    
    val_set = ShapeNetDataset(cfg)

    cfg = {"save_model_path": "model_weights/shapenet_airplane_model_weights",
           'show_model_summary': True, 
           'npoints': 2500,
           'in_dim': 6,
           'train': {"epochs": 10, 'lr': 0.05, 
                     'weight_decay': 1e-4, 'momentum':0.9, 
                     'save_checkpoint_interval': 5,
                     'train_subset': False, # set False if not intending to use subset. Set to 20 or something for small dataset experimentation/debugging
                     'val_subset': False, # see above
                     'num_classes': 4, # 4 due to only training on airplane
                     'continue_training': False,
                     'weights_path': "model_weights/model_weights_segmentation_4.pt"} 
            }

    segmentation_trainer = PTSegmentationTrainer(train_set = train_set, val_set = val_set, cfg = cfg)
    segmentation_trainer.train_segmentation()