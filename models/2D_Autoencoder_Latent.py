import os
import numpy as np
from glob import glob
import pandas as pd
from monai.transforms import (
    LoadImage, EnsureChannelFirst, ScaleIntensity, EnsureType, Compose, ToTensor
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import AutoEncoder
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torchvision

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
        return {"image": image, "slice_label": 0}  

class CustomAutoEncoder(AutoEncoder):
    def __init__(self, spatial_dims, in_channels, out_channels, channels, strides):
        super().__init__(spatial_dims, in_channels, out_channels, channels, strides)
    
    def semantic_encoder(self, x):
        for layer in self.encode:
            x = layer(x)
        return x

def extract_latent_spaces(model, dataloader, device):
    latent_spaces = []
    model.eval()
    with torch.no_grad():
        for batch_data in dataloader:
            inputs = batch_data["image"].to(device).float()  
            latents = model.semantic_encoder(inputs)
            print("Latent space shape:", latents.shape)
            print("Latent space values:", latents.cpu().numpy())
            latent_spaces.append(latents.cpu().numpy())
    return np.concatenate(latent_spaces, axis=0)

def manipulate_latent_spaces(latent_spaces, direction, alpha=0.1):
    return latent_spaces + alpha * direction

def decode_and_visualize(model, original_latent_spaces, manipulated_latent_spaces, device, original_images, s):
    model.eval()
    with torch.no_grad():
        original_latents = torch.tensor(original_latent_spaces, dtype=torch.float32).to(device)
        manipulated_latents = torch.tensor(manipulated_latent_spaces, dtype=torch.float32).to(device)
        noise = torch.randn_like(original_images).to(device)
        reconstruction = model.decode(original_latents).to(device)
        manipulated_reconstruction = model.decode(manipulated_latents).to(device)

    nb = 8
    original_images = original_images[:nb].to(device)
    grid = torchvision.utils.make_grid(torch.cat([original_images, reconstruction[:nb], manipulated_reconstruction[:nb]]), 
                                       nrow=8, normalize=False, scale_each=False, pad_value=0)
    plt.figure(figsize=(15,5))
    plt.imshow(grid.permute(1, 2, 0).detach().cpu().numpy(), cmap= 'gray')
    plt.axis('off')
    plt.title(f"Original, Reconstruction, Manipulated s = {s}")
    plt.show()

def load_subject_images(subjects, base_dir="./output_slices_preprocessed"):
    image_files = []
    for subject in subjects:
        subject_dir = os.path.join(base_dir, subject)
        image_files.extend(glob(os.path.join(subject_dir, "*.png")))
    return image_files

eval_transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(), 
    ScaleIntensity(), 
    EnsureType(),  
    ToTensor() 
])

def evaluate_and_manipulate_autoencoder():
    data_info = pd.read_csv('Cleaned_Lipo_Data.csv')
    
    benign_subjects = data_info[data_info['Diagnosis_binary'] == 0]['Subject'].tolist()
    malignant_subjects = data_info[data_info['Diagnosis_binary'] == 1]['Subject'].tolist()
    
    benign_images = load_subject_images(benign_subjects)
    malignant_images = load_subject_images(malignant_subjects)
    
    benign_ds = ImageDataset(image_files=benign_images, transform=eval_transforms)
    malignant_ds = ImageDataset(image_files=malignant_images, transform=eval_transforms)
    benign_loader = DataLoader(benign_ds, batch_size=16, shuffle=False, num_workers=0)
    malignant_loader = DataLoader(malignant_ds, batch_size=16, shuffle=False, num_workers=0)
    
    train_loader = DataLoader(benign_ds + malignant_ds, batch_size=16, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomAutoEncoder(spatial_dims=2, in_channels=1, out_channels=1, channels=(16, 32, 64), strides=(2, 2, 2)).to(device)
    
    # Load the best trained model
    model.load_state_dict(torch.load("best_autoencoder.pth"))
    
    latents_train = extract_latent_spaces(model, train_loader, device)
    latents_val = extract_latent_spaces(model, benign_loader, device)

    print("Latent spaces (train) shape:", latents_train.shape)
    print("Latent spaces (train) values:", latents_train[:5])
    
    print("Latent spaces (validation) shape:", latents_val.shape)
    print("Latent spaces (validation) values:", latents_val[:5])
    
    labels_train = np.array([0] * len(benign_images) + [1] * len(malignant_images))
    
    latents_train_flat = latents_train.reshape(latents_train.shape[0], -1)
    latents_val_flat = latents_val.reshape(latents_val.shape[0], -1)
    
    scaler = StandardScaler()
    latents_train_flat = scaler.fit_transform(latents_train_flat)
    latents_val_flat = scaler.transform(latents_val_flat)
    
    clf = LogisticRegression(max_iter=10000) 
    clf.fit(latents_train_flat, labels_train)
    
    manipulation_direction = clf.coef_.reshape(latents_train.shape[1:])
    s = 100.0
    
    manipulated_latent_spaces = manipulate_latent_spaces(latents_val, manipulation_direction, alpha=s)
    
    original_images = next(iter(benign_loader))["image"].to(device)
    decode_and_visualize(model, latents_val, manipulated_latent_spaces, device, original_images, s)

if __name__ == "__main__":
    evaluate_and_manipulate_autoencoder()
