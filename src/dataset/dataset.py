from monai.transforms import Compose, RandCropByLabelClassesd, RandRotate90d, RandFlipd
from monai.data import Dataset, DataLoader, CacheDataset
import os
import numpy as np

def create_inference_dataloader(img_dir, label_dir, non_random_transforms=None, random_transforms=None, batch_size=16, num_workers=4):
    data =[]
    image = np.load(img_dir)
    data.append({"image": image})
    
    ds = CacheDataset(data=data, transform=non_random_transforms, cache_rate=1.0)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return loader    

def create_dataloaders(train_img_dir,
                       train_label_dir, 
                       val_img_dir, 
                       val_label_dir,
                       non_random_transforms=None,
                       random_transforms=None,
                       batch_size=16,
                       num_workers=4
                       ):
    
    # output batch = batch_size * num_samples : if random_transforms has RandCropByLabelClassesd
    
    train_list = os.listdir(train_img_dir)
    val_list = os.listdir(val_img_dir)
    train_files = []
    valid_files = []
    for name in train_list:
        image = np.load(os.path.join(train_label_dir, f"{name}"))    
        label = np.load(os.path.join(train_label_dir, f"{name.replace("image", "label")}"))

        train_files.append({"image": image, "label": label})    

    for name in val_list:
        image = np.load(os.path.join(val_img_dir, f"{name}"))
        label = np.load(os.path.join(val_label_dir, f"{name.replace("image", "label")}"))

        valid_files.append({"image": image, "label": label})

    train_ds = CacheDataset(data=train_files, transform=non_random_transforms, cache_rate=1.0)
    train_ds = Dataset(data=train_ds, transform=random_transforms)
    
    val_ds = CacheDataset(data=valid_files, transform=non_random_transforms, cache_rate=1.0)
    val_ds = Dataset(data=val_ds, transform=random_transforms)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

