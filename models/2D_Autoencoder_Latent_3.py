import os
import numpy as np
import pandas as pd
from glob import glob
from monai.transforms import (
    LoadImage, EnsureChannelFirst, ScaleIntensity, EnsureType, Compose, ToTensor,
    RandRotate, RandFlip, RandZoom
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import AutoEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import torchvision

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

class SupervisedAutoEncoder(nn.Module):
    def __init__(self, spatial_dims=2, in_channels=1, out_channels=1, channels=(16, 32, 64), strides=(2, 2, 2)):
        super(SupervisedAutoEncoder, self).__init__()
        
        self.autoencoder = AutoEncoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 256, 256)  
            dummy_output = self.autoencoder.encode(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  
        )

    def forward(self, x):
        encoded = self.autoencoder.encode(x)  
        z_flattened = encoded.view(encoded.size(0), -1)  
        classification = self.classifier(z_flattened) 

        reconstruction = self.autoencoder.decode(encoded) 
        return reconstruction, classification

def train_supervised_autoencoder(model, dataloader, val_loader, device, epochs=20, patience=5):
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    reconstruction_loss_fn = nn.MSELoss()
    classification_loss_fn = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    trigger_times = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss, total_reconstruction_loss, total_classification_loss = 0, 0, 0
        
        for batch_data in dataloader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            
            optimizer.zero_grad()
            
            reconstruction, classification = model(inputs)
            
            reconstruction_loss = reconstruction_loss_fn(reconstruction, inputs)
            classification_loss = classification_loss_fn(classification, labels)
            
            loss = reconstruction_loss + classification_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_reconstruction_loss += reconstruction_loss.item()
            total_classification_loss += classification_loss.item()
        
        val_loss = evaluate_classification_performance(model, val_loader, device, return_loss=True)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, '
              f'Reconstruction Loss: {total_reconstruction_loss:.4f}, '
              f'Classification Loss: {total_classification_loss:.4f}, '
              f'Validation Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
        else:
            trigger_times += 1
        
        if trigger_times >= patience:
            print("Early stopping triggered")
            break

def evaluate_classification_performance(model, dataloader, device, return_loss=False):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    reconstruction_loss_fn = nn.MSELoss()
    classification_loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_data in dataloader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            reconstruction, classification = model(inputs)
            probs = nn.functional.softmax(classification, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:,1].cpu().numpy())  
            
            if return_loss:
                reconstruction_loss = reconstruction_loss_fn(reconstruction, inputs)
                classification_loss = classification_loss_fn(classification, labels)
                total_loss += (reconstruction_loss + classification_loss).item()
    
    acc = accuracy_score(all_labels, all_preds)
    print(f'Accuracy: {acc:.4f}')
    
    print('Classification Report:')
    print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant']))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    
    if return_loss:
        return total_loss

def extract_latent_spaces(model, dataloader, device):
    model.eval()
    latent_spaces = []
    labels = []

    with torch.no_grad():
        for batch_data in dataloader:
            inputs = batch_data['image'].to(device)
            batch_labels = batch_data['label'].to(device)
            
            latents = model.autoencoder.encode(inputs)
            
            latent_spaces.append(latents.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())

    latent_spaces = np.concatenate(latent_spaces, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    return latent_spaces, labels

def plot_latent_space_heatmap(latent_spaces, labels, method='pca'):
    n_samples, n_channels, height, width = latent_spaces.shape
    latent_spaces_flat = latent_spaces.reshape(n_samples, n_channels * height * width)
    
    if method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    latent_2d = reducer.fit_transform(latent_spaces_flat)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Labels')
    plt.title(f'Latent Space ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

def train_logistic_regression_on_latent_space(latent_spaces, labels):
    latent_spaces_flat = latent_spaces.reshape(latent_spaces.shape[0], -1)
    
    
    clf = LogisticRegression(max_iter=10000)
    clf.fit(latent_spaces_flat, labels)
    
    return clf

def manipulate_latent_space(latent_space, clf, direction='towards_benign', alpha=500.0):
    coef = clf.coef_.reshape(latent_space.shape)
    coef = coef / np.linalg.norm(coef)  
    coef = torch.tensor(coef, dtype=torch.float32, device=latent_space.device)
    
    if direction == 'towards_malignant':
        manipulated_latent = latent_space + alpha * coef
    elif direction == 'towards_benign':
        manipulated_latent = latent_space - alpha * coef
    else:
        raise ValueError("Direction must be either 'towards_malignant' or 'towards_benign'")
    
    return manipulated_latent

def visualize_latent_space_manipulation(model, original_latents, manipulated_latents, device, original_images):
    model.eval()
    
    with torch.no_grad():
        original_reconstructions = model.autoencoder.decode(original_latents.to(device))
        manipulated_reconstructions = model.autoencoder.decode(manipulated_latents.to(device))
    
    original_images = original_images.to(device)
    
    nb = min(8, original_images.size(0)) 
    
    grid = torch.cat([original_images[:nb], original_reconstructions[:nb], manipulated_reconstructions[:nb]])
    grid = torchvision.utils.make_grid(grid, nrow=nb, normalize=True, pad_value=0)
    
    plt.figure(figsize=(15, 5))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title("Original, Reconstruction, Manipulated")
    plt.axis("off")
    plt.show()

def visualize_reconstructions(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for batch_data in dataloader:
            inputs = batch_data["image"].to(device)
            reconstruction, _ = model(inputs)  
            inputs = inputs.cpu().numpy()
            reconstruction = reconstruction.cpu().numpy()

            fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(20, 5))
            for i in range(8):
                axes[0, i].imshow(inputs[i, 0], cmap="gray")
                axes[0, i].axis("off")
                axes[0, i].set_title('Original')
                axes[1, i].imshow(reconstruction[i, 0], cmap="gray")
                axes[1, i].axis("off")
                axes[1, i].set_title('Reconstructed')

            plt.show()
            break 

def main():
    data_info = pd.read_csv('Cleaned_Lipo_Data.csv')
    
    subject_to_label = dict(zip(data_info['Subject'], data_info['Diagnosis_binary']))

    eval_transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(), 
        ScaleIntensity(), 
        RandRotate(range_x=15, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        EnsureType(),  
        ToTensor() 
    ])

    data_dir = "./output_slices_preprocessed"
    all_subjects = list(subject_to_label.keys())
    all_images = []
    all_labels = []

    for subject in all_subjects:
        subject_dir = os.path.join(data_dir, subject)
        subject_images = glob(os.path.join(subject_dir, "*.png"))
        all_images.extend(subject_images)
        all_labels.extend([subject_to_label[subject]] * len(subject_images))

    train_images, val_images, train_labels, val_labels = train_test_split(
        all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )

    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_labels = torch.tensor(val_labels, dtype=torch.long)

    train_ds = ImageDataset(image_files=train_images, labels=train_labels, transform=eval_transforms)
    val_ds = ImageDataset(image_files=val_images, labels=val_labels, transform=eval_transforms)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SupervisedAutoEncoder().to(device)

    train_supervised_autoencoder(model, train_loader, val_loader, device, epochs=20)

    print("\nClassification Performance on Validation Set:")
    evaluate_classification_performance(model, val_loader, device)

    latent_spaces, labels = extract_latent_spaces(model, val_loader, device)

    # Optionally save the latent spaces for later use
    np.save("latent_spaces.npy", latent_spaces)
    np.save("latent_labels.npy", labels)
    print(f"Latent spaces shape: {latent_spaces.shape}")
    print(f"Labels shape: {labels.shape}")

    plot_latent_space_heatmap(latent_spaces, labels, method='pca')

    print("\nReconstruction Results on Validation Set:")
    visualize_reconstructions(model, val_loader, device)

    clf = train_logistic_regression_on_latent_space(latent_spaces, labels)

    print("Logistic Regression Coefficients:", clf.coef_)

    idx = 0  
    original_latent = torch.tensor(latent_spaces[idx]).unsqueeze(0).to(device)
    original_image = val_loader.dataset[idx]['image'].clone().detach().unsqueeze(0).to(device)

    # Manipulate the latent space
    manipulated_latent = manipulate_latent_space(original_latent, clf, direction='towards_benign', alpha=500.0)

    # Visualize the manipulation
    visualize_latent_space_manipulation(model, original_latent, manipulated_latent, device, original_image)

if __name__ == "__main__":
    main()

