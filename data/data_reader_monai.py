import os
import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    Lambdad,
    EnsureChannelFirstd,
    ScaleIntensityd,
    CenterSpatialCropd,
    Resized,
    MapTransform,
)
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, root_dir, section, transform=None):
        self.root_dir = root_dir
        self.section = section
        self.transform = transform
        self.image_paths = self._load_images()

    def _load_images(self):
        section_dir = os.path.join(self.root_dir, self.section)
        if not os.path.exists(section_dir):
            raise FileNotFoundError(f"Directory not found: {section_dir}")
        
        image_paths = []
        for sub_dir in os.listdir(section_dir):
            sub_dir_path = os.path.join(section_dir, sub_dir)
            if os.path.isdir(sub_dir_path):
                image_path = os.path.join(sub_dir_path, "image.nii.gz")
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                else:
                    print(f"Image not found: {image_path}")
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        sample = {"image": image_path}

        if self.transform:
            sample = self.transform(sample)

        return sample

class DynamicCenterSpatialCropd(MapTransform):
    def __init__(self, keys, roi_size_3d, roi_size_2d):
        super().__init__(keys)
        self.roi_size_3d = roi_size_3d
        self.roi_size_2d = roi_size_2d

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image_shape = d[key].shape
            print(f"Image shape before cropping: {image_shape}")
            if len(image_shape) == 4:
                cropper = CenterSpatialCropd(keys=[key], roi_size=self.roi_size_3d)
            else:
                cropper = CenterSpatialCropd(keys=[key], roi_size=self.roi_size_2d)
            d = cropper(d)
        return d

data_transform = Compose(
    [
        LoadImaged(keys=["image"]),
        Lambdad(keys="image", func=lambda x: x[:, :, :, 1] if x.ndim == 4 else x[:, :, 1]),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        ScaleIntensityd(keys=["image"]),
        DynamicCenterSpatialCropd(keys=["image"], roi_size_3d=(160, 200, 155), roi_size_2d=(160, 200)),
        Resized(keys=["image"], spatial_size=(32, 40, 32)),
    ]
)

root_dir = os.path.abspath("Data/worc")

train_ds = CustomDataset(root_dir=root_dir, section="training", transform=data_transform)
val_ds = CustomDataset(root_dir=root_dir, section="validation", transform=data_transform)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=8, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=8, persistent_workers=True)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i in range(4):
    sample = train_ds[i * 20]
    image_data = sample["image"]
    for j in range(2):
        slice_index = j * 10 + 5
        slice_data = image_data[0, :, :, slice_index].detach().cpu().numpy()
        axes[j, i].imshow(slice_data, vmin=0, vmax=1, cmap="gray")
        axes[j, i].set_title(f"Image {i*20}, Slice {slice_index}")
        axes[j, i].axis("off")

plt.tight_layout()
plt.show()
