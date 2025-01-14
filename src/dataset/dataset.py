from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    Orientationd, CropForegroundd, GaussianSmoothd, ScaleIntensityd,
    RandSpatialCropd, RandRotate90d, RandFlipd, RandGaussianSmoothd,
    ToTensord, RandCropByLabelClassesd
)

from monai.data import Dataset, DataLoader, CacheDataset
import os
import numpy as np
import torch
from torch.utils.data import Subset

def create_inference_dataloader(img_dir, label_dir, non_random_transforms=None, random_transforms=None, batch_size=16, num_workers=4):
    data = []
    image = np.load(img_dir)
    data.append({"image": image})
    
    ds = CacheDataset(data=data, transform=non_random_transforms, cache_rate=1.0)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return loader    

def make_dataset(img_dir, label_dir, non_random_transforms=None, random_transforms=None):
    files = []
    img_list = [f for f in os.listdir(img_dir) if f.endswith('.npy')]
    for name in img_list:
        image = np.load(os.path.join(img_dir, f"{name}"))
        label = np.load(os.path.join(label_dir, f"{name}"))

        files.append({"image": image, "label": label})
        
    ds = CacheDataset(data=files, transform=non_random_transforms, cache_rate=1.0)
    ds = Dataset(data=ds, transform=random_transforms)
    
    return ds

def make_dataloader(img_dir, label_dir, non_random_transforms=None, random_transforms=None, batch_size=16, num_workers=4,num_repeat=3):
    
    ds = make_dataset(img_dir, label_dir, non_random_transforms, random_transforms)
    indices = torch.arange(len(ds)).repeat(num_repeat)  # 각 데이터를 3번 호출
    subset = Subset(ds, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return loader

def create_dataloaders(train_img_dir,
                       train_label_dir, 
                       val_img_dir, 
                       val_label_dir,
                       non_random_transforms=None,
                       random_transforms=None,
                       batch_size=16,
                       num_workers=4,
                        train_num_repeat=1
                       ):
    
    train_loader = make_dataloader(train_img_dir, train_label_dir, non_random_transforms, random_transforms, batch_size, num_workers,train_num_repeat)
    val_loader = make_dataloader(val_img_dir, val_label_dir, non_random_transforms, random_transforms, batch_size, num_workers)
    
    return train_loader, val_loader

def create_dataloaders_bw(train_img_dir,
                       train_label_dir, 
                       val_img_dir, 
                       val_label_dir,
                       non_random_transforms=None,
                       val_non_random_transforms=None,
                       random_transforms=None,
                       batch_size=16,
                       num_workers=4,
                       train_num_repeat=1
                       ):
    
    train_loader = make_dataloader(train_img_dir, train_label_dir, non_random_transforms, random_transforms, batch_size, num_workers,train_num_repeat)
    val_loader = make_dataloader(val_img_dir, val_label_dir, val_non_random_transforms, random_transforms, batch_size, num_workers)
    
    return train_loader, val_loader

if __name__ ==  "__main__":
    
    train_img_dir = "./datasets/train/images"
    train_label_dir = "./datasets/train/labels"
    val_img_dir = "./datasets/val/images"
    val_label_dir = "./datasets/val/labels"
    loader_batch = 2
    
    non_random_transforms = Compose([
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS")
    ])
    
    random_transforms = Compose([
        GaussianSmoothd(
        keys=["image"],      # 변환을 적용할 키
        sigma=[1.0, 1.0, 1.0]  # 각 축(x, y, z)의 시그마 값
        ),
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=[96, 96, 96],
            num_classes=7,
            num_samples=8  # num_samples 값을 양의 정수로 설정
        ),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[1,2]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),    
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),    
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),    
    ])
    
    train_loader, val_loader = create_dataloaders(train_img_dir, 
                                                  train_label_dir, 
                                                  val_img_dir, 
                                                  val_label_dir, 
                                                  non_random_transforms, 
                                                  random_transforms, 
                                                  loader_batch
                                                  )

    batch = next(iter(train_loader))
    images, labels = batch["image"], batch["label"]
    print(images.shape, labels.shape)