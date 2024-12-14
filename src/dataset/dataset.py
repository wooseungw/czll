from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    
)

def calculate_patch_starts(dimension_size: int, patch_size: int) -> List[int]:
    """최소 오버랩으로 패치 시작 위치 계산"""
    if dimension_size <= patch_size:
        return [0]
    
    n_patches = np.ceil(dimension_size / patch_size)
    total_overlap = (n_patches * patch_size - dimension_size) / (n_patches - 1)
    
    positions = []
    for i in range(int(n_patches)):
        pos = int(i * (patch_size - total_overlap))
        if pos + patch_size > dimension_size:
            pos = dimension_size - patch_size
        if pos not in positions:
            positions.append(pos)
    
    return positions

class CryoET_2_5D_Dataset(Dataset):
    def __init__(self, 
                 data_dir: str, 
                 mode: str = 'train',
                 slice_depth: int = 3, 
                 crop_size: Tuple[int, int] = (96, 96),
                 transform=None):
        
        self.data_dir = Path(data_dir) / mode
        self.slice_depth = slice_depth
        self.crop_size = crop_size
        self.transform = transform
        self.pad_size = slice_depth // 2
        self.mode = mode

        self.image_files = sorted(list((self.data_dir / 'images').glob('*.npy')))
        
        # 파일 shape 정보 저장
        self.file_shapes = {}
        for img_file in self.image_files:
            mmap_array = np.load(img_file, mmap_mode='r')
            self.file_shapes[img_file] = mmap_array.shape
        
        # train/val 모드에서 라벨 파일 처리
        if self.mode in ['train', 'val']:
            self.label_files = sorted(list((self.data_dir / 'labels').glob('*.npy')))
            if len(self.image_files) != len(self.label_files):
                raise ValueError(f"이미지 파일과 라벨 파일 개수가 다릅니다: {len(self.image_files)} vs {len(self.label_files)}")
        else:
            self.label_files = None
        
        # 패치 인덱스 계산
        self.patch_indices = []
        for file_idx, image_file in enumerate(self.image_files):
            shape = self.file_shapes[image_file]
            D, H, W = shape
            
            z_starts = range(0, D - slice_depth + 1)
            y_starts = calculate_patch_starts(H, crop_size[0])
            x_starts = calculate_patch_starts(W, crop_size[1])
            
            for z in z_starts:
                for y in y_starts:
                    for x in x_starts:
                        self.patch_indices.append({
                            'file_idx': file_idx,
                            'z_start': z,
                            'y_start': y,
                            'x_start': x
                        })
    
    def __len__(self):
        return len(self.patch_indices)
    
    def __getitem__(self, idx):
        info = self.patch_indices[idx]
        file_idx = info['file_idx']
        z_start = info['z_start']
        y_start = info['y_start']
        x_start = info['x_start']
        
        # memmap으로 데이터 로드
        image_mmap = np.load(self.image_files[file_idx], mmap_mode='r')
        
        # 패치 추출 (복사본 생성)
        image_patch = np.array(image_mmap[
            z_start:z_start + self.slice_depth,
            y_start:y_start + self.crop_size[0],
            x_start:x_start + self.crop_size[1]
        ])
        
        data_dict = {
            'image': image_patch[None, ...],  # (1, D, H, W)
            'crop_info': {
                'file_idx': file_idx,
                'z_start': z_start,
                'z_center': z_start + self.slice_depth // 2,
                'y_start': y_start,
                'x_start': x_start,
                'crop_size': self.crop_size,
                'file_name': self.image_files[file_idx].stem
            }
        }
        
        # train/val 모드에서만 라벨 처리
        if self.mode in ['train', 'val']:
            label_mmap = np.load(self.label_files[file_idx], mmap_mode='r')
            center_z = z_start + self.slice_depth // 2
            label_patch = np.array(label_mmap[
                center_z,
                y_start:y_start + self.crop_size[0],
                x_start:x_start + self.crop_size[1]
            ])
            data_dict['label'] = label_patch[None, ...]  # (1, H, W)
        
        if self.transform:
            data_dict = self.transform(data_dict)
        
        return data_dict

# Transform 설정
train_transforms = Compose([
    EnsureChannelFirstd(
        keys=['image', 'label'],
        channel_dim=0,
        allow_missing_keys=True
    ),
    NormalizeIntensityd(
        keys=['image'],
        allow_missing_keys=True
    )
])

def create_dataloaders(data_dir: str, batch_size: int = 8, crop_size: int = 96, num_workers: int = 0):
    train_ds = CryoET_2_5D_Dataset(
        data_dir=data_dir,
        mode='train',
        slice_depth=3,
        crop_size=(crop_size, crop_size),
        transform=train_transforms
    )
    
    val_ds = CryoET_2_5D_Dataset(
        data_dir=data_dir,
        mode='val',
        slice_depth=3,
        crop_size=(crop_size, crop_size),
        transform=None
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


# import time
# from tqdm import tqdm

# if __name__ == "__main__":
#     data_dir = "./datasets"
#     crop_size = 126
#     batch_size = 256

#     # 테스트할 num_workers 값 리스트
#     num_workers_list = [0, 2, 4, 8]
#     num_batches_to_test = 10  # 테스트할 배치 수 제한

#     for num_workers in tqdm(num_workers_list, desc="Testing num_workers"):
#         print(f"\nTesting DataLoader with num_workers={num_workers}...")
        
#         # DataLoader 생성
#         start_time = time.time()
#         train_loader, val_loader = create_dataloaders(
#             data_dir=data_dir, 
#             batch_size=batch_size, 
#             crop_size=crop_size, 
#             num_workers=num_workers
#         )
#         loader_creation_time = time.time() - start_time
#         print(f"DataLoader 생성 시간: {loader_creation_time:.2f}초")

        # # 배치 로드 시간 측정
        # start_time = time.time()
        # for idx, batch in enumerate(train_loader):
        #     if idx >= num_batches_to_test:
        #         break
        # loading_time = time.time() - start_time
        # print(f"num_workers={num_workers} | {num_batches_to_test}개 배치 로드 시간: {loading_time:.2f}초")
        
        # batch = next(iter(train_loader))
        # print("이미지 shape:", batch['image'].shape)
        # print("라벨 shape:", batch['label'].shape)
        # print("Crop 정보:", batch['crop_info'])

