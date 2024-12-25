import os
import numpy as np
from glob import glob
from monai.transforms import (
    LoadImage, EnsureChannelFirst, ScaleIntensity, EnsureType, Compose, ToTensor
)
from monai.data import Dataset, DataLoader
import torch
from torch.nn import MSELoss
from torch.optim import Adam
import torch.nn as nn
import torchvision.models as models
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Custom Dataset class for loading images
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

# Custom AutoEncoder class using ResNet18 as the encoder
class ResNetAutoEncoder(torch.nn.Module):
    def __init__(self, input_shape=(1, 256, 256)):
        super().__init__()
        # Use ResNet18 as the encoder, modifying the first conv layer for single-channel input
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the ResNet's fully connected layer and adaptive pool to keep spatial dimensions
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        
        # Pass a dummy input through the encoder to determine the size of the output
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.encoder(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)
        
        # Define a new fully connected layer for latent representation
        self.fc = nn.Linear(flattened_size, 128 * 16 * 16)  # Adjusted output to match decoder input

        # Define the decoder part
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Use Sigmoid to keep output in range [0, 1]
        )

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        
        # Decoder: Reshape to match the expected input for the decoder
        x = x.view(x.size(0), 128, 16, 16)  # Reshape to [batch_size, 128, 16, 16]
        x = self.decoder(x)
        return x

# Function to extract the latent space from the encoder
def extract_latent_space(model, data_loader, device):
    model.eval()
    latents = []
    with torch.no_grad():
        for data in data_loader:
            inputs = data.to(device)
            latent = model.encoder(inputs)
            latents.append(latent.cpu().numpy())
    return np.concatenate(latents, axis=0)

# Main function to train the autoencoder, apply logistic regression, and manipulate the latent space
def train_autoencoder():
    train_transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(), 
        ScaleIntensity(), 
        EnsureType(),  
        ToTensor() 
    ])

    # Specify the directory containing preprocessed image slices
    data_dir = "./output_slices_preprocessed"
    all_images = glob(os.path.join(data_dir, "*/*.png"))

    print(f"Found {len(all_images)} images.")

    np.random.shuffle(all_images)
    train_files = all_images[:int(0.8 * len(all_images))]
    val_files = all_images[int(0.8 * len(all_images)):]

    train_ds = ImageDataset(image_files=train_files, transform=train_transforms)
    val_ds = ImageDataset(image_files=val_files, transform=train_transforms)

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")

    # Reduce batch size to prevent GPU overload
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetAutoEncoder(input_shape=(1, 256, 256)).to(device)

    loss_function = MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    num_epochs = 50
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, batch_data in enumerate(train_loader):
            inputs = batch_data.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

            # Log progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            # Free unused memory
            torch.cuda.empty_cache()

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs = val_data.to(device)
                val_outputs = model(val_inputs)
                val_loss += loss_function(val_outputs, val_inputs).item() * val_inputs.size(0)

                # Free unused memory
                torch.cuda.empty_cache()
                
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_autoencoder.pth")
            print("Saved Best Model")

    model.load_state_dict(torch.load("best_autoencoder.pth"))

    # Extract latent space
    latents_train = extract_latent_space(model, train_loader, device)
    latents_val = extract_latent_space(model, val_loader, device)

    # Placeholder: Replace with your actual labels for the logistic regression
    train_labels = np.zeros(latents_train.shape[0])
    val_labels = np.zeros(latents_val.shape[0])

    # Train logistic regression on the latent space
    clf = LogisticRegression(solver="newton-cg", random_state=0).fit(latents_train, train_labels)
    print(f"Train accuracy: {clf.score(latents_train, train_labels)}")
    print(f"Validation accuracy: {clf.score(latents_val, val_labels)}")

    w = torch.Tensor(clf.coef_).float().to(device)

    # Manipulate the latent space and reconstruct images
    s = -1.5
    model.eval()
    with torch.no_grad():
        for batch_data in val_loader:
            inputs = batch_data.to(device)
            latent = model.encoder(inputs)

            # Manipulate latent space
            latent_manip = latent + s * w[0]

            # Reconstruct images
            outputs = model.decoder(latent.unsqueeze(2).unsqueeze(3))
            outputs_manip = model.decoder(latent_manip.unsqueeze(2).unsqueeze(3))

            inputs = inputs.cpu().numpy()
            outputs = outputs.cpu().numpy()
            outputs_manip = outputs_manip.cpu().numpy()

            fig, axes = plt.subplots(nrows=3, ncols=8, figsize=(20, 8))
            for i in range(8):
                axes[0, i].imshow(inputs[i, 0], cmap="gray")
                axes[0, i].axis("off")
                axes[0, i].set_title('Original')
                axes[1, i].imshow(outputs[i, 0], cmap="gray")
                axes[1, i].axis("off")
                axes[1, i].set_title('Reconstructed')
                axes[2, i].imshow(outputs_manip[i, 0], cmap="gray")
                axes[2, i].axis("off")
                axes[2, i].set_title('Manipulated')

            plt.show()
            break

if __name__ == "__main__":
    train_autoencoder()
