import torch
import torch.nn as nn
import torchvision.models as models
from monai.networks.nets import AutoEncoder

class CustomAutoEncoderWithSemantic(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, channels, strides, embedding_dimension=64):
        super(CustomAutoEncoderWithSemantic, self).__init__()
        
        # Autoencoder from MONAI
        self.autoencoder = AutoEncoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
        )
        
        # Semantic Encoder using ResNet
        resnet = models.resnet18(pretrained=True)
        self.semantic_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),  # Adapted for single channel
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, embedding_dimension)
        )
        
        # Fully connected layer to combine the semantic encoder output with autoencoder latent space
        self.fc = nn.Linear(embedding_dimension + channels[-1] * (256 // (2 ** len(strides)))**2, channels[-1] * (256 // (2 ** len(strides)))**2)

    def forward(self, x):
        # Get latent representation from the semantic encoder
        latent_semantic = self.semantic_encoder(x)
        
        # Use the autoencoder's encoder
        encoded = self.autoencoder.encode(x)
        
        # Flatten the encoded representation for concatenation
        encoded_flat = encoded.view(encoded.size(0), -1)
        
        # Combine both representations and reshape for decoding
        combined_latent = torch.cat((encoded_flat, latent_semantic), dim=1)
        combined_latent = self.fc(combined_latent)
        combined_latent = combined_latent.view(encoded.size(0), encoded.size(1), encoded.size(2), encoded.size(3))
        
        # Reconstruct the image using combined latent representations
        reconstruction = self.autoencoder.decode(combined_latent)
        return reconstruction, latent_semantic

# Example usage
if __name__ == "__main__":
    # Example input size of (batch_size, channels, height, width) -> (16, 1, 256, 256)
    x = torch.randn(16, 1, 256, 256)
    model = CustomAutoEncoderWithSemantic(spatial_dims=2, in_channels=1, out_channels=1, channels=(16, 32, 64), strides=(2, 2, 2), embedding_dimension=64)
    
    # Forward pass
    reconstruction, latent_semantic = model(x)
    print("Reconstruction shape:", reconstruction.shape)
    print("Latent semantic shape:", latent_semantic.shape)



