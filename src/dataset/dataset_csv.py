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

import pandas as pd
import os

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_train_val_csv(
    dataset_dir,
    original_csv_path,
    train_csv_path,
    val_csv_path,
    test_size=0.2,
    random_state=42,
    file_extension=".npy"
):
    """
    Create train.csv and val.csv from dataset directory.

    Args:
        dataset_dir (str): The root directory containing 'images' and 'labels'.
        train_csv_path (str): Output path for train CSV.
        val_csv_path (str): Output path for validation CSV.
        test_size (float): Proportion of data to use for validation.
        random_state (int): Seed for reproducibility.
        file_extension (str): File extension for data files (default: '.npy').
    """
    image_dir = os.path.join(dataset_dir, "images")
    label_dir = os.path.join(dataset_dir, "labels")

    # Ensure directories exist
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        raise ValueError("Image or label directory does not exist in the dataset folder.")

    # Gather all files in the image directory
    img_files = [f for f in os.listdir(image_dir) if f.endswith(file_extension)]
    data = []

    for img_file in img_files:
        image_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, img_file)
        
        # Check if label file exists
        if not os.path.exists(label_path):
            print(f"Warning: Label file missing for {img_file}. Skipping.")
            continue
        
        data.append({"image": image_path, "label": label_path})

    # Split into train and validation sets
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=random_state)

    # Save to CSV
    pd.DataFrame(data).to_csv(original_csv_path, index=False)
    pd.DataFrame(train_data).to_csv(train_csv_path, index=False)
    pd.DataFrame(val_data).to_csv(val_csv_path, index=False)

    print(f"Train CSV created at: {data}")
    print(f"Train CSV created at: {train_csv_path}")
    print(f"Validation CSV created at: {val_csv_path}")

def create_inference_dataloader(img_dir, label_dir, non_random_transforms=None, random_transforms=None, batch_size=16, num_workers=4):
    data = []
    image = np.load(img_dir)
    data.append({"image": image})
    
    ds = CacheDataset(data=data, transform=non_random_transforms, cache_rate=0.9)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return loader    

def make_dataset_from_csv(csv_file, non_random_transforms=None, random_transforms=None):
    import pandas as pd
    from monai.data import Dataset, CacheDataset
    import numpy as np
    
    # CSV 읽기
    df = pd.read_csv(csv_file)
    
    # 경로에서 데이터를 읽어와 CacheDataset에 적합한 형식으로 변환
    files = []
    for _, row in df.iterrows():
        files.append({
            "image": np.load(row["image"]),  # CSV에서 'image' 경로로 .npy 파일 로드
            "label": np.load(row["label"])   # CSV에서 'label' 경로로 .npy 파일 로드
        })

    # CacheDataset 생성
    ds = CacheDataset(data=files, transform=non_random_transforms, cache_rate=1.0)
    ds = Dataset(data=ds, transform=random_transforms)
    
    return ds

def make_dataloader_from_csv(csv_file, non_random_transforms=None, random_transforms=None, batch_size=16, num_workers=4, num_repeat=3):
    ds = make_dataset_from_csv(csv_file, non_random_transforms, random_transforms)
    indices = torch.arange(len(ds)).repeat(num_repeat)  # 각 데이터를 num_repeat번 호출
    subset = Subset(ds, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return loader

def create_dataloaders_from_csv(train_csv, 
                                val_csv, 
                                non_random_transforms=None,
                                random_transforms=None,
                                batch_size=16,
                                num_workers=4,
                                train_num_repeat=3):
    train_loader = make_dataloader_from_csv(train_csv, non_random_transforms, random_transforms, batch_size, num_workers, train_num_repeat)
    val_loader = make_dataloader_from_csv(val_csv, non_random_transforms, random_transforms, batch_size, num_workers)
    
    return train_loader, val_loader

if __name__ ==  "__main__":
    
    train_img_dir = "./datasets/train/images"
    train_label_dir = "./datasets/train/labels"
    val_img_dir = "./datasets/val/images"
    val_label_dir = "./datasets/val/labels"
    train_csv = "./datasets/train.csv"
    val_csv = "./datasets/val.csv"

    loader_batch = 2
    
    non_random_transforms = Compose([
    EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
    NormalizeIntensityd(keys="image"),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    GaussianSmoothd(
        keys=["image"],      # 변환을 적용할 키
        sigma=[1.0, 1.0, 1.0]  # 각 축(x, y, z)의 시그마 값
        ),
    ])
    random_transforms = Compose([
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=[96,96,96],
            num_classes=7,
            num_samples=2, 
            
        ),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[1, 2]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    ])

    # 데이터 로더 생성
    train_loader, val_loader = create_dataloaders_from_csv(
        train_csv=train_csv, 
        val_csv=val_csv, 
        non_random_transforms=non_random_transforms,
        random_transforms=random_transforms,
        batch_size=loader_batch,
        num_workers=0
        
    )

    batch = next(iter(train_loader))
    images, labels = batch["image"], batch["label"]
    print(images.shape, labels.shape)