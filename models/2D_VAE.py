import os
from glob import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from monai.transforms import (
    ScaleIntensity, EnsureType, Compose, ToTensor
)

# Custom Dataset class to handle your directory structure
class CustomDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_paths = glob(os.path.join(self.root_dir, 'Lipo-*/*.png'))  # Adjust the extension if necessary

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = plt.imread(image_path)  # Read as numpy array
        if self.transforms:
            image = self.transforms(image)
        return image

# Define transforms
transforms = Compose([
    lambda x: np.expand_dims(x, axis=0),  # Add channel dimension manually
    ScaleIntensity(),  # Scale intensity to [0, 1]
    ToTensor(),  # Convert to tensor
    EnsureType(),  # Ensure correct tensor type
])

# Load dataset
dataset = CustomDataset(root_dir='output_slices_preprocessed', transforms=transforms)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define the VAE Model
class VAE(nn.Module):
    def __init__(self, input_shape=(1, 256, 256), latent_dim=32):
        super(VAE, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(128 * 32 * 32, latent_dim)
        self.fc_logvar = nn.Linear(128 * 32 * 32, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 128 * 32 * 32)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 128, 32, 32)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar

# Instantiate the model
vae = VAE(input_shape=(1, 256, 256), latent_dim=32)
vae.to(torch.float32)

# Define the VAE loss function
def vae_loss(reconstructed_x, x, mu, logvar):
    reconstruction_loss = F.mse_loss(reconstructed_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence

# Training setup
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    vae.train()
    train_loss = 0
    for batch in dataloader:
        batch = batch.to(torch.float32)
        optimizer.zero_grad()
        reconstructed_batch, mu, logvar = vae(batch)
        loss = vae_loss(reconstructed_batch, batch, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f"Epoch {epoch + 1}, Loss: {train_loss / len(dataloader.dataset)}")

# Save the model
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, "vae_model.pth")
torch.save(vae.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Visualization
vae.eval()
with torch.no_grad():
    # Get a batch of original images
    original_images = next(iter(dataloader)).to(torch.float32)
    # Reconstruct the images
    reconstructed_images, _, _ = vae(original_images)
    # Generate new images by sampling from the latent space
    z = torch.randn(16, 32).to(torch.float32)  # Generate 16 random samples from the latent space
    generated_images = vae.decode(z)

# Plotting
def plot_images(original, reconstructed, generated, n=8):
    plt.figure(figsize=(18, 6))
    for i in range(n):
        # Original images
        plt.subplot(3, n, i + 1)
        plt.imshow(original[i][0], cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Reconstructed images
        plt.subplot(3, n, i + 1 + n)
        plt.imshow(reconstructed[i][0], cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')

        # Generated images
        plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(generated[i][0], cmap='gray')
        plt.title("Generated")
        plt.axis('off')
    
    plt.show()

# Visualize the results
plot_images(original_images.cpu(), reconstructed_images.cpu(), generated_images.cpu())
