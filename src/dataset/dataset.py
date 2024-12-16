from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    Orientationd, CropForegroundd, GaussianSmoothd, ScaleIntensityd,
    RandSpatialCropd, RandRotate90d, RandFlipd, RandGaussianNoised,
    ToTensord, RandCropByLabelClassesd
)
from monai.data import Dataset, DataLoader, CacheDataset
import os
import numpy as np

def create_inference_dataloader(img_dir, label_dir, non_random_transforms=None, random_transforms=None, batch_size=16, num_workers=4):
    data = []
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
    
    train_list = os.listdir(train_img_dir)
    val_list = os.listdir(val_img_dir)
    train_files = []
    valid_files = []
    for name in train_list:
        image = np.load(os.path.join(train_img_dir, f"{name}"))    
        label = np.load(os.path.join(train_label_dir, f"{name}"))

        train_files.append({"image": image, "label": label})    

    for name in val_list:
        image = np.load(os.path.join(val_img_dir, f"{name}"))
        label = np.load(os.path.join(val_label_dir, f"{name}"))

        valid_files.append({"image": image, "label": label})

    train_ds = CacheDataset(data=train_files, transform=non_random_transforms, cache_rate=1.0)
    train_ds = Dataset(data=train_ds, transform=random_transforms)
    
    val_ds = CacheDataset(data=valid_files, transform=non_random_transforms, cache_rate=1.0)
    val_ds = Dataset(data=val_ds, transform=random_transforms)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
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
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=[3, 96, 96],
            num_classes=7,
            num_samples=8  # num_samples 값을 양의 정수로 설정
        ),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[1,2]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),    
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