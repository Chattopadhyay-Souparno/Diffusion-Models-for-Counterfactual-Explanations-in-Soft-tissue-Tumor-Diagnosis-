import os
import numpy as np
from glob import glob
from monai.transforms import (
    LoadImage, EnsureChannelFirst, ScaleIntensity, EnsureType, Compose, ToTensor
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import AutoEncoder
import torch
import matplotlib.pyplot as plt

class ImageDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = self.image_files[idx]
        if self.transform:
            image = self.transform(image)
        return image

def evaluate_autoencoder():
  
    eval_transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(), 
        ScaleIntensity(), 
        EnsureType(),  
        ToTensor() 
    ])

    data_dir = "./output_slices_preprocessed"
    all_images = glob(os.path.join(data_dir, "*/*.png"))

    eval_files = all_images

    eval_ds = ImageDataset(image_files=eval_files, transform=eval_transforms)
    eval_loader = DataLoader(eval_ds, batch_size=16, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder(spatial_dims=2, in_channels=1, out_channels=1, channels=(16, 32, 64), strides=(2, 2, 2)).to(device)

    model.load_state_dict(torch.load("best_autoencoder.pth"))

    model.eval()
    with torch.no_grad():
        for batch_data in eval_loader:
            inputs = batch_data.to(device)
            outputs = model(inputs)
            inputs = inputs.cpu().numpy()
            outputs = outputs.cpu().numpy()

            fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(20, 5))
            for i in range(8):
                axes[0, i].imshow(inputs[i, 0], cmap="gray")
                axes[0, i].axis("off")
                axes[0, i].set_title('Original')
                axes[1, i].imshow(outputs[i, 0], cmap="gray")
                axes[1, i].axis("off")
                axes[1, i].set_title('Reconstructed')

            plt.show()
            break

if __name__ == "__main__":
    evaluate_autoencoder()
