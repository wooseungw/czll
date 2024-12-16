import os
import shutil
import copick
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import numpy as np

class Preprocessor:
    def __init__(self, config_blob, copick_config_path="./kaggle/working/copick.config"):
        self.config_blob = config_blob
        self.copick_config_path = copick_config_path
        # 클래스 내부 메서드 호출 시 self 사용
        self.copick_config_path = self._make_config()
        self.root = copick.from_file(self.copick_config_path)

    def _make_config(self, static_root="./kaggle/input/czii-cryo-et-object-identification/train/static"):
        config_blob = f"""{{
        "name": "czii_cryoet_mlchallenge_2024",
        "description": "2024 CZII CryoET ML Challenge training data.",
        "version": "1.0.0",

        "pickable_objects": [
            {{
                "name": "apo-ferritin",
                "is_particle": true,
                "pdb_id": "4V1W",
                "label": 1,
                "color": [  0, 117, 220, 128],
                "radius": 60,
                "map_threshold": 0.0418
            }},
            {{
              "name" : "beta-amylase",
                "is_particle": true,
                "pdb_id": "8ZRZ",
                "label": 2,
                "color": [255, 255, 255, 128],
                "radius": 90,
                "map_threshold": 0.0578  
            }},
            {{
                "name": "beta-galactosidase",
                "is_particle": true,
                "pdb_id": "6X1Q",
                "label": 3,
                "color": [ 76,   0,  92, 128],
                "radius": 90,
                "map_threshold": 0.0578
            }},
            {{
                "name": "ribosome",
                "is_particle": true,
                "pdb_id": "6EK0",
                "label": 4,
                "color": [  0,  92,  49, 128],
                "radius": 150,
                "map_threshold": 0.0374
            }},
            {{
                "name": "thyroglobulin",
                "is_particle": true,
                "pdb_id": "6SCJ",
                "label": 5,
                "color": [ 43, 206,  72, 128],
                "radius": 130,
                "map_threshold": 0.0278
            }},
            {{
                "name": "virus-like-particle",
                "is_particle": true,
                "label": 6,
                "color": [255, 204, 153, 128],
                "radius": 135,
                "map_threshold": 0.201
            }},
            {{
                "name": "membrane",
                "is_particle": false,
                "label": 8,
                "color": [100, 100, 100, 128]
            }},
            {{
                "name": "background",
                "is_particle": false,
                "label": 9,
                "color": [10, 150, 200, 128]
            }}
        ],

        "overlay_root": "./kaggle/working/overlay",

        "overlay_fs_args": {{
            "auto_mkdir": true
        }},

        "static_root": "{static_root}"
        }}"""
        with open(self.copick_config_path, "w") as f:
            f.write(config_blob)
        print(f"Config file written to {self.copick_config_path}")
        return self.copick_config_path

    def processing(self, run, voxel_size=10, tomo_type="denoised", task="train"):
        tomogram = run.get_voxel_spacing(voxel_size).get_tomogram(tomo_type).numpy()
        if task == "train":
            copick_segmentation_name = "paintedPicks"
            copick_user_name = "copickUtils"
            segmentation = run.get_segmentations(
                name=copick_segmentation_name,
                user_id=copick_user_name,
                voxel_size=voxel_size,
                is_multilabel=True
            )[0].numpy()
            return tomogram, segmentation
        else:
            return tomogram
            

    def processing_train(self, voxel_size=10):
        tomo_type_list = ["ctfdeconvolved", "denoised", "isonetcorrected", "wbp"]
        # Define directories for saving numpy arrays
        train_image_dir = Path('./datasets/train/images')
        train_label_dir = Path('./datasets/train/labels')
        val_image_dir = Path('./datasets/val/images')
        val_label_dir = Path('./datasets/val/labels')

        for dir_path in [train_image_dir, train_label_dir, val_image_dir, val_label_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # root.runs의 개수 출력
        print(f"Number of runs: {len(self.root.runs)}")

        for tomo_type in tomo_type_list:
            print(f"Processing \"{tomo_type}\" data...")
            for vol_idx, run in enumerate(self.root.runs):
                # Load image and label data (task="train"으로 segmentation 포함)
                tomogram, segmentation = self.processing(run, voxel_size, tomo_type, task="train")

                # Format run name
                run_name = run.name.replace("\\", "_").replace("/", "_")

                # Determine if this is the last volume
                is_last_volume = (vol_idx == len(self.root.runs) - 1)

                # Set directories based on whether it's the last volume
                image_dir = val_image_dir if is_last_volume else train_image_dir
                label_dir = val_label_dir if is_last_volume else train_label_dir

                # Save tomogram and segmentation as numpy arrays
                image_path = image_dir / f"{tomo_type}_{run_name}.npy"
                label_path = label_dir / f"{tomo_type}_{run_name}.npy"
                np.save(image_path, tomogram)
                np.save(label_path, segmentation)

                # 저장된 파일 경로 출력
                print(f"Saved image: {image_path}")
                print(f"Saved label: {label_path}")

        # 저장된 파일의 개수 출력
        print(f"Number of files in train images: {len(list(train_image_dir.glob('*.npy')))}")
        print(f"Number of files in train labels: {len(list(train_label_dir.glob('*.npy')))}")
        print(f"Number of files in val images: {len(list(val_image_dir.glob('*.npy')))}")
        print(f"Number of files in val labels: {len(list(val_label_dir.glob('*.npy')))}")

        print("Processing complete.")

if __name__ == "__main__":
    config_blob = """{
    "name": "czii_cryoet_mlchallenge_2024",
    "description": "2024 CZII CryoET ML Challenge training data.",
    "version": "1.0.0",

    "pickable_objects": [
        {
            "name": "apo-ferritin",
            "is_particle": true,
            "pdb_id": "4V1W",
            "label": 1,
            "color": [  0, 117, 220, 128],
            "radius": 60,
            "map_threshold": 0.0418
        },
        {
          "name" : "beta-amylase",
            "is_particle": true,
            "pdb_id": "8ZRZ",
            "label": 2,
            "color": [255, 255, 255, 128],
            "radius": 90,
            "map_threshold": 0.0578  
        },
        {
            "name": "beta-galactosidase",
            "is_particle": true,
            "pdb_id": "6X1Q",
            "label": 3,
            "color": [ 76,   0,  92, 128],
            "radius": 90,
            "map_threshold": 0.0578
        },
        {
            "name": "ribosome",
            "is_particle": true,
            "pdb_id": "6EK0",
            "label": 4,
            "color": [  0,  92,  49, 128],
            "radius": 150,
            "map_threshold": 0.0374
        },
        {
            "name": "thyroglobulin",
            "is_particle": true,
            "pdb_id": "6SCJ",
            "label": 5,
            "color": [ 43, 206,  72, 128],
            "radius": 130,
            "map_threshold": 0.0278
        },
        {
            "name": "virus-like-particle",
            "is_particle": true,
            "label": 6,
            "color": [255, 204, 153, 128],
            "radius": 135,
            "map_threshold": 0.201
        },
        {
            "name": "membrane",
            "is_particle": false,
            "label": 8,
            "color": [100, 100, 100, 128]
        },
        {
            "name": "background",
            "is_particle": false,
            "label": 9,
            "color": [10, 150, 200, 128]
        }
    ],

    "overlay_root": "./kaggle/working/overlay",

    "overlay_fs_args": {
        "auto_mkdir": true
    },

    "static_root": "./kaggle/input/czii-cryo-et-object-identification/train/static"
    }"""

    preprocessor = Preprocessor(config_blob)
    preprocessor.processing_train()