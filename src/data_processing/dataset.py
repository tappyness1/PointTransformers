import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor, PILToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
from src.data_processing.dataset_utils import farthest_point_sample, pc_normalize
from src.constants import exclude_list
import json
import numpy as np


class ModelNetDataset(Dataset):
    def __init__(self, cfg):
        self.data_path = cfg['data_path']
        if cfg['train']:
            self.dataset_path = self.data_path + f"/{cfg['modelnet_type']}_train.txt"
        else:
            self.dataset_path = self.data_path + f"/{cfg['modelnet_type']}_test.txt"
        
        self.dataset = [line.rstrip() for line in open(self.dataset_path)]

        shape_names = [line.rstrip() for line in open(self.data_path + f"/{cfg['modelnet_type']}_shape_names.txt")]
        self.classes = dict(zip(shape_names, range(len(shape_names))))
        # self.classes = {shape: i for i, shape in enumerate(shape_names)}
        self.npoints = cfg['npoints']
        
    def __len__(self): 
        return len(self.dataset)
    
    def __getitem__(self, idx):

        item_name = self.dataset[idx].split('_')
        class_name = item_name[0] if len(item_name) == 2 else '_'.join(item_name[:-1])
        point_cloud_path = os.path.join(self.data_path,class_name, f"{self.dataset[idx]}.txt")
        point_clouds = pd.read_csv(point_cloud_path, header = None).to_numpy().astype('float32')
        point_clouds = farthest_point_sample(point_clouds, self.npoints) # can use torch3d but this is fine too
        point_clouds = pc_normalize(point_clouds)
        class_id = self.classes[class_name]

        return point_clouds[:, :3], class_id
    
class ShapeNetDataset(Dataset):

    def __init__(self, cfg):
        self.data_path = cfg['data_path']
        if cfg['train']:
            self.dataset_path = os.path.join(self.data_path, "train_test_split", "shuffled_train_file_list.json")
        else:
            self.dataset_path = os.path.join(self.data_path, "train_test_split", "shuffled_val_file_list.json")

        f = open(self.dataset_path)
        data = json.load(f)
        data_split = [instance.split("/") for instance in data]

        # only get the instances of the class. eg airplane only instances
        # self.dataset = ["/".join(instance[1:]) for instance in data_split if instance[1] == cfg['instance']]
        self.dataset = []
        for instance in data_split:  
            if instance[1] == cfg['instance']:
                instance_path = "/".join(instance[1:])
                # point_cloud_path = os.path.join(self.data_path, f"{instance_path}.txt")
                # point_clouds = pd.read_csv(point_cloud_path, 
                #             delimiter=" ", 
                #             names = ["x", "y", "z", "r", "g", "b", "label"], 
                #             header=None)
                # if point_clouds.shape[0] == 0:
                #   print (instance_path)
                #   continue
                if instance_path in exclude_list:
                  continue
                self.dataset.append("/".join(instance[1:]))


        self.cfg = cfg

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        point_cloud_path = os.path.join(self.data_path, f"{self.dataset[idx]}.txt")

        point_clouds = pd.read_csv(point_cloud_path, 
                                   delimiter=" ", 
                                   names = ["x", "y", "z", "r", "g", "b", "label"], 
                                   header=None).to_numpy().astype('float32')
        
        # if the we have less points than the cut, then sample from the existing points to make up for it
        if point_clouds.shape[0] < self.cfg['shape_cut_off']:
            # print (point_clouds.shape)
            # print (self.dataset[idx])
            delta_points = self.cfg['shape_cut_off'] - point_clouds.shape[0]
            dummy_points_index = np.random.randint(0, point_clouds.shape[0], delta_points)
            point_clouds = np.concatenate([point_clouds, point_clouds[dummy_points_index]], axis=0)

        point_clouds = point_clouds[:self.cfg['shape_cut_off']]

        # TODO: control the dummy points by introducing mask

        return point_clouds[:, :3], point_clouds[:, -1]
    
        

if __name__ == "__main__":
    # cfg = {"data_path": "../data/modelnet40_normal_resampled", "train": False, "modelnet_type": "modelnet10"}
    # model40net_dataset = ModelNetDataset(cfg)
    # dataloader = torch.utils.data.DataLoader(model40net_dataset, batch_size=4, shuffle=True)
    # for i, (point_clouds, class_id) in enumerate(dataloader):
    #     print (point_clouds.shape, class_id.shape)
    #     break

    cfg = {"data_path": "../data/shapenetcore_partanno_segmentation_benchmark_v0_normal", 
           "train": True, 
           "instance": "02691156",
           "shape_cut_off": 2500}
    shapenet_dataset = ShapeNetDataset(cfg)
    dataloader = torch.utils.data.DataLoader(shapenet_dataset, batch_size=10, shuffle=True)
    for i, (point_clouds, class_id) in enumerate(dataloader):
        print (point_clouds.shape, class_id.shape)
        break
