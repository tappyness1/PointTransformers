from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import os 
import torch
from torchvision import transforms 
from torch.utils.data import DataLoader


class PSNet(Dataset):
    def __init__(self, annotations_dir: str):
        self.annotations_dir = Path(annotations_dir)
        self.annotations = None

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        transform = transforms.Compose([transforms.PILToTensor(),
                                        transforms.Resize((512, 512))])
        mask = transform(Image.open(os.path.join(self.mask_dir, self.masks[idx]))).to(torch.float32)
        img = transform(Image.open(os.path.join(self.images_dir, self.images[idx]))).to(torch.float32)
        img /= 255
        mask /= 255
        return [img, mask]

if __name__ == "__main__":
    annotations_dir = "../data/PSNet5/"
    dataset = PSNet(annotations_dir)
    train_dataloader = DataLoader(dataset, batch_size=6, shuffle = True)
    img, mask = next(iter(train_dataloader))
    print (mask.unique())
