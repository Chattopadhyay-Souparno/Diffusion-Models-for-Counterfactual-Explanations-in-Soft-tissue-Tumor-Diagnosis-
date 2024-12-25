import os
import numpy as np
import pandas as pd
from glob import glob
from monai.transforms import (
    LoadImage, EnsureChannelFirst, ScaleIntensity, EnsureType, Compose, ToTensor, Resize
)
from monai.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# Define a custom dataset
class ImageDataset(Dataset):
    def __init__(self, image_files, labels, transform=None):
        self.image_files = image_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = self.image_files[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return {"image": image, "label": label}

# Define the ResNet-based AutoEncoder
class ResNetAutoEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(ResNetAutoEncoder, self).__init__()

        # Load pre-trained ResNet model
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),  # Modified for 1 input channel
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        # Adaptive pooling to get a fixed-size output
        self.pool = nn.AdaptiveAvgPool2d((8, 8))

        # Decoder to reconstruct the image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, out_channels, kernel_size=3, stride=2, output_padding=1),
            nn.Sigmoid()  # Use sigmoid to map the output to [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.pool(z)
        reconstruction = self.decoder(z)
        # Resize the output to match the input image size
        reconstruction = nn.functional.interpolate(reconstruction, size=(256, 256), mode='bilinear', align_corners=False)
        return reconstruction

# Training function with validation and early stopping
def train_autoencoder(model, train_loader, val_loader, device, epochs=50, patience=10):
    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    reconstruction_loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_data in train_loader:
            inputs = batch_data["image"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstruction = model(inputs)
            
            # Calculate loss
            loss = reconstruction_loss_fn(reconstruction, inputs)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate validation loss
        val_loss = evaluate_autoencoder(model, val_loader, device, reconstruction_loss_fn)
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {total_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_autoencoder_resnet.pth')  # Save the best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

# Evaluation function for calculating validation loss
def evaluate_autoencoder(model, val_loader, device, loss_fn):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_data in val_loader:
            inputs = batch_data["image"].to(device)
            reconstruction = model(inputs)
            loss = loss_fn(reconstruction, inputs)
            val_loss += loss.item()
    return val_loss

# Function to visualize reconstructions
def visualize_reconstructions(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for batch_data in dataloader:
            inputs = batch_data["image"].to(device)
            reconstruction = model(inputs)
            inputs = inputs.cpu().numpy()
            reconstruction = reconstruction.cpu().numpy()

            # Plot original and reconstructed images side by side
            fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(20, 5))
            for i in range(8):
                axes[0, i].imshow(inputs[i, 0], cmap="gray")
                axes[0, i].axis("off")
                axes[0, i].set_title('Original')
                axes[1, i].imshow(reconstruction[i, 0], cmap="gray")
                axes[1, i].axis("off")
                axes[1, i].set_title('Reconstructed')

            plt.show()
            break  # Only show one batch of images

# Main function to load data, train, and evaluate the model
def main():
    # Load the CSV file with subject information
    data_info = pd.read_csv('Cleaned_Lipo_Data.csv')
    
    # Create a mapping from subject name to label
    subject_to_label = dict(zip(data_info['Subject'], data_info['Diagnosis_binary']))

    # Define transforms
    eval_transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(), 
        ScaleIntensity(), 
        Resize((256, 256)),  # Ensure all images are resized to 256x256
        EnsureType(),  
        ToTensor() 
    ])

    # Data directory and image files
    data_dir = "./output_slices_preprocessed"
    all_subjects = list(subject_to_label.keys())
    all_images = []
    all_labels = []

    # Map images to their labels based on the subject folders
    for subject in all_subjects:
        subject_dir = os.path.join(data_dir, subject)
        subject_images = glob(os.path.join(subject_dir, "*.png"))
        all_images.extend(subject_images)
        all_labels.extend([subject_to_label[subject]] * len(subject_images))

    # Convert labels to tensor
    labels = torch.tensor(all_labels, dtype=torch.long)

    # Split into training and validation sets
    train_size = int(0.8 * len(all_images))
    val_size = len(all_images) - train_size
    train_images, val_images = torch.utils.data.random_split(all_images, [train_size, val_size])

    # Create datasets and dataloaders
    train_ds = ImageDataset(image_files=[all_images[i] for i in train_images.indices], labels=[labels[i] for i in train_images.indices], transform=eval_transforms)
    val_ds = ImageDataset(image_files=[all_images[i] for i in val_images.indices], labels=[labels[i] for i in val_images.indices], transform=eval_transforms)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetAutoEncoder().to(device)

    # Train the model
    train_autoencoder(model, train_loader, val_loader, device, epochs=200)

    # Visualize reconstruction results
    print("\nReconstruction Results:")
    visualize_reconstructions(model, val_loader, device)

if __name__ == "__main__":
    main()
