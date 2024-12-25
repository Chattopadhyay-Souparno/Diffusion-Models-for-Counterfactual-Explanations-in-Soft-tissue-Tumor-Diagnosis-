import os
import numpy as np
from glob import glob
from monai.transforms import (
    LoadImage, EnsureChannelFirst, ScaleIntensity, EnsureType, Compose, ToTensor
)
from monai.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import torchvision
import time
import sys
import matplotlib.pyplot as plt
from generative.inferers import DiffusionInferer
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from generative.networks.schedulers.ddim import DDIMScheduler


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


class Diffusion_AE(torch.nn.Module):
    def __init__(self, embedding_dimension=64):
        super().__init__()
        self.unet = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(128, 256, 256),
            attention_levels=(True, True, True),
            num_res_blocks=1,
            num_head_channels=64,
            with_conditioning=True,
            cross_attention_dim=1,
        )
        self.semantic_encoder = torchvision.models.resnet18(pretrained=False)
        self.semantic_encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.semantic_encoder.fc = torch.nn.Linear(512, embedding_dimension)

    def forward(self, xt, x_cond, t):
        latent = self.semantic_encoder(x_cond)
        noise_pred = self.unet(x=xt, timesteps=t, context=latent.unsqueeze(2))
        return noise_pred, latent


def train_diffusion_autoencoder():
    train_transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        EnsureType(),
        ToTensor()
    ])

    data_dir = "./output_patches_preprocessed"
    all_images = glob(os.path.join(data_dir, "*/*.png"))

    print(f"Found {len(all_images)} images.")

    np.random.shuffle(all_images)
    train_files = all_images[:int(0.8 * len(all_images))]
    val_files = all_images[int(0.8 * len(all_images)):]

    train_ds = ImageDataset(image_files=train_files, transform=train_transforms)
    val_ds = ImageDataset(image_files=val_files, transform=train_transforms)

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Diffusion_AE(embedding_dimension=64).to(device)
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)
    inferer = DiffusionInferer(scheduler)

    n_iterations = int(1e4)
    batch_size = 64
    val_interval = 100
    iter_loss_list, val_iter_loss_list = [], []
    iterations = []
    iteration, iter_loss = 0, 0

    total_start = time.time()

    while iteration < n_iterations:
        for batch in train_loader:
            iteration += 1
            model.train()
            optimizer.zero_grad(set_to_none=True)
            images = batch.to(device)
            noise = torch.randn_like(images).to(device)
            timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (batch_size,)).to(device).long()
            latent = model.semantic_encoder(images)
            noise_pred = inferer(inputs=images, diffusion_model=model.unet, noise=noise, timesteps=timesteps, condition=latent.unsqueeze(2))
            loss = F.mse_loss(noise_pred.float(), noise.float())

            loss.backward()
            optimizer.step()

            iter_loss += loss.item()
            sys.stdout.write(f"Iteration {iteration}/{n_iterations} - train Loss {loss.item():.4f}" + "\r")
            sys.stdout.flush()

            if iteration % val_interval == 0:
                model.eval()
                val_iter_loss = 0
                for val_step, val_batch in enumerate(val_loader):
                    with torch.no_grad():
                        images = val_batch.to(device)
                        timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (batch_size,)).to(device).long()
                        noise = torch.randn_like(images).to(device)
                        latent = model.semantic_encoder(images)
                        noise_pred = inferer(inputs=images, diffusion_model=model.unet, noise=noise, timesteps=timesteps, condition=latent.unsqueeze(2))
                        val_loss = F.mse_loss(noise_pred.float(), noise.float())

                    val_iter_loss += val_loss.item()
                iter_loss_list.append(iter_loss / val_interval)
                val_iter_loss_list.append(val_iter_loss / (val_step + 1))
                iterations.append(iteration)
                iter_loss = 0
                print(f"Iteration {iteration} - Interval Loss {iter_loss_list[-1]:.4f}, Interval Loss Val {val_iter_loss_list[-1]:.4f}")

    total_time = time.time() - total_start
    print(f"Train diffusion completed, total time: {total_time}.")

    plt.style.use("seaborn-bright")
    plt.title("Learning Curves Diffusion Model", fontsize=20)
    plt.plot(iterations, iter_loss_list, color="C0", linewidth=2.0, label="Train")
    plt.plot(iterations, val_iter_loss_list, color="C4", linewidth=2.0, label="Validation")
    plt.yticks(fontsize=12), plt.xticks(fontsize=12)
    plt.xlabel("Iterations", fontsize=16), plt.ylabel("Loss", fontsize=16)
    plt.legend(prop={"size": 14})
    plt.show()

    # Save the trained model
    torch.save(model.state_dict(), "best_diffusion_autoencoder.pth")

    # Load and evaluate the best model
    model.load_state_dict(torch.load("best_diffusion_autoencoder.pth"))
    model.eval()
    with torch.no_grad():
        for batch_data in val_loader:
            inputs = batch_data.to(device)
            noise = torch.randn_like(inputs).to(device)
            timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (batch_size,)).to(device).long()
            latent = model.semantic_encoder(inputs)
            reconstructions = inferer(inputs=inputs, diffusion_model=model.unet, noise=noise, timesteps=timesteps, condition=latent.unsqueeze(2))
            inputs = inputs.cpu().numpy()
            reconstructions = reconstructions.cpu().numpy()

            fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(20, 5))
            for i in range(8):
                axes[0, i].imshow(inputs[i, 0], cmap="gray")
                axes[0, i].axis("off")
                axes[0, i].set_title('Original')
                axes[1, i].imshow(reconstructions[i, 0], cmap="gray")
                axes[1, i].axis("off")
                axes[1, i].set_title('Reconstructed')

            plt.show()
            break


if __name__ == "__main__":
    train_diffusion_autoencoder()
