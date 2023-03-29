import torchvision
import torch
import pytorch_lightning as pl 

from torchvision.datasets import MNIST
from torchvision import transforms
from typing import Optional


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = '../data/MNIST', batch_size: int = 128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transforms = transforms.Compose([
            transforms.ToTensor(),                        # [0, 255] -> [0.0, 1.0]
            transforms.Normalize(mean=(0.5), std=(0.5)),  # [0.0, 1.0] -> [-1.0, 1.0]      
            ])
    
    def prepare_data(self):
        torchvision.datasets.MNIST(self.data_dir, train=True, download=True)
        torchvision.datasets.MNIST(self.data_dir, train=False, download=True)
    
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage in None:
            self.train_data = torchvision.datasets.MNIST(self.data_dir, train=True, transform=self.transforms)
        
        if stage == "test":
            self.test_data =  torchvision.datasets.MNIST(self.data_dir, train=False, transform=self.transforms)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,drop_last=True ,num_workers=0)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, drop_last=True ,num_workers=0)

