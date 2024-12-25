
# **Diffusion Models for Counterfactual Explanations in Soft-tissue Tumor Diagnosis**

## **Overview**
This project, developed under **Erasmus MC, Rotterdam**, focuses on the reconstruction and counterfactual generation of medical images using diffusion models, latent space analysis, and various autoencoder architectures. The aim is to create interpretable and effective solutions for soft-tissue tumor diagnosis through advanced generative models.

---

## **Key Features**
1. **Data Handling**:
   - Download, preprocess, and prepare medical datasets for model training.
   - Convert volumetric NIfTI files into manageable 2D slices or patches.

2. **Model Architectures**:
   - **Basic Autoencoders**: Focus on reconstruction with simple encoder-decoder structures.
   - **ResNet-based Autoencoders**: Utilize ResNet encoders for richer feature extraction.
   - **Diffusion Autoencoders**: Combine MONAI and diffusion-based techniques for denoising and latent regularization.
   - **Variational Autoencoders (VAEs)**: Use probabilistic latent spaces for counterfactual generation.

3. **Latent Space Analysis**:
   - Analyze and manipulate learned representations for semantic feature extraction and counterfactual explanation generation.

4. **Visualization**:
   - Visualize reconstruction and counterfactual results alongside latent space analysis outputs.

---

## **Repository Structure**

### **Root Level**
```
.
├── data/                             # Data preparation and preprocessing scripts
├── models/                           # Model architectures
├── training/                         # Training scripts for various models
├── evaluation/                       # Evaluation and visualization scripts
├── scripts/                          # Utility and experimental scripts
├── LICENSE.txt                       # Project license information
├── .gitignore                        # Ignore unnecessary files in the repository
└── README.md                         # Project documentation
```

---

### **Directory and File Descriptions**

#### **1. data/**
Scripts for downloading, preparing, and preprocessing datasets.
- `datadownloader.py`: Downloads datasets (e.g., WORC) from the XNAT repository.
- `data_patch_preprocessing.py`: Normalizes, resizes, and preprocesses image patches.
- `data_reader.py`: Converts 3D NIfTI images into 2D slices.
- `data_reader_monai.py`: MONAI-compatible data reader for medical images.
- `data_reader_patches.py`: Converts NIfTI images into 2D patches.
- `diffusion_autoencoder_monai.py`: Implements diffusion autoencoder pipeline using MONAI.

---

#### **2. models/**
Contains various model architectures used in the project.
- `2D_Autoencoder.py`: Basic 2D autoencoder for image reconstruction.
- `2D_Autoencoder_Latent.py`: Extracts and manipulates latent representations.
- `2D_Autoencoder_Latent_2.py`: Variant for latent space analysis.
- `2D_Autoencoder_Latent_3.py`: Another latent space variant.
- `2D_Semantic_Resnet.py`: Semantic ResNet-based autoencoder for feature-rich representations.
- `2D_VAE.py`: Implements variational autoencoder for generative modeling.
- `Resnet_Autoencoder.py`: Combines ResNet encoder with a custom decoder.
- `Resnet_Autoencoder_2.py`: Variant of ResNet-based autoencoder.
- `Resnet_MONAI.py`: Integrates MONAI and ResNet into a unified autoencoder.

---

#### **3. training/**
Scripts to train models and manage experiments.
- `monai_train.py`: Trains basic MONAI autoencoder with defined transforms and evaluation setup.

---

#### **4. evaluation/**
Tools for evaluation and visualization.
- `Image reconstruction.pptx`: Summary of reconstruction results and findings.
- `metrics.py`: Scripts for evaluating reconstruction quality using metrics like MSE, PSNR, and SSIM (to be added).

---

#### **5. scripts/**
General-purpose utility and experimental scripts.
- `automlcomparison.py`: Compares AutoML configurations.
- `default_experiments.py`: Contains default experiment setups.
- `gif_maker.py`: Generates GIFs from reconstruction results.
- `utils.py`: Utility functions for various tasks.

---

## **Installation**

### **Prerequisites**
- Python 3.8 or higher.
- CUDA-enabled GPU (recommended for faster training).

### **Dependencies**
Install the required packages:
```bash
pip install -r requirements.txt
```

---

## **Usage**

### **1. Data Preparation**
- Download the dataset:
  ```bash
  python data/datadownloader.py
  ```
- Preprocess the dataset:
  ```bash
  python data/data_reader.py
  python data/data_patch_preprocessing.py
  ```

### **2. Model Training**
- Train any model by running its corresponding script:
  ```bash
  python models/2D_Autoencoder.py
  ```

### **3. Evaluation**
- Evaluate reconstructions or visualize results:
  ```bash
  python evaluation/visualize_results.py
  ```

---

## **Results**
- High-quality reconstructions have been achieved using:
  - Basic Autoencoders: Focused on minimizing reconstruction errors.
  - ResNet-based Autoencoders: Enhanced semantic feature extraction.
  - Diffusion Autoencoders: Improved denoising and robustness.

- Latent space analysis highlights:
  - Clear separability of semantic features.
  - Potential for generating counterfactual explanations.

---

## **Acknowledgments**
This project was developed at **Erasmus MC, Rotterdam**, under the guidance of the **Biomedical Imaging Group Rotterdam (BIGR)**.

For questions or suggestions, feel free to reach out or open an issue in the repository.
