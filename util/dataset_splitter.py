import os
import torch
import numpy as np

class DatasetSplitter:
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.base_path = f"dataset/{dataset_name}/processed/splits"
        os.makedirs(self.base_path, exist_ok=True)
    
    def save_split(self, retain_data, forget_data, split_name="default"):
        split_dir = os.path.join(self.base_path, split_name)
        os.makedirs(split_dir, exist_ok=True)
        

        torch.save(retain_data, os.path.join(split_dir, "retain_data.pt"))
        torch.save(forget_data, os.path.join(split_dir, "forget_data.pt"))
        
    
    def load_split(self, split_name="default"):
        split_dir = os.path.join(self.base_path, split_name)
        
        if not os.path.exists(split_dir):
            raise ValueError(f"Split '{split_name}' does not exist for dataset {self.dataset_name}")
        
        retain_data = torch.load(os.path.join(split_dir, "retain_data.pt"))
        forget_data = torch.load(os.path.join(split_dir, "forget_data.pt"))
        
        return retain_data, forget_data
    
    def list_splits(self):
        if not os.path.exists(self.base_path):
            return []
        
        return [d for d in os.listdir(self.base_path) 
                if os.path.isdir(os.path.join(self.base_path, d))]
