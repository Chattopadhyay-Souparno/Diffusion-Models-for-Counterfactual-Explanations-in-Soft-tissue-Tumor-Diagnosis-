import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Global variables
input_path = './output_slices/mask'
output_path = './output_masks_preprocessed'

# Ensure the output base folder exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

def load_png_image(file_path):
    return np.array(Image.open(file_path))

def save_png_image(image_array, save_path):
    Image.fromarray(image_array).convert("L").save(save_path)

def normalize_image(image):
    # Normalize the image to the range [0, 255]
    image = image.astype(np.float32)
    image -= image.min()
    if image.max() > 0:
        image /= image.max()
    image *= 255
    return image.astype(np.uint8)

from skimage.transform import resize

def resample_image_2d(image, new_shape):
    """
    Resample a 2D image to a new shape.

    :param image: Input 2D image as a NumPy array.
    :param new_shape: Tuple indicating the new shape (new_height, new_width).
    :return: Resampled 2D image.
    """
    resampled_image = resize(image, new_shape, preserve_range=True, anti_aliasing=True)
    return resampled_image.astype(image.dtype)


def preprocess_and_save_png_image(image_path, save_dir, count, new_shape):
    image = load_png_image(image_path)
    
    # Resampling the image
    resampled_image = resample_image_2d(image, new_shape)
    
    # Normalizing the image
    norm_image = normalize_image(resampled_image)
    
    # Saving the normalized image
    save_path = os.path.join(save_dir, f'image_{count:03d}.png')
    save_png_image(norm_image, save_path)
    return save_path

def main():
    # Target new shape for the images, e.g., (256, 256)
    new_shape = (256, 256)
    
    # Get the list of subdirectories in the input path
    subdirectories = [os.path.join(input_path, sub_dir) for sub_dir in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, sub_dir))]
    
    count = 1
    for sub_dir in tqdm(subdirectories, desc="Processing directories"):
        image_files = [os.path.join(sub_dir, file) for file in os.listdir(sub_dir) if file.endswith('.png')]
        
        output_sub_dir = os.path.join(output_path, os.path.basename(sub_dir))
        if not os.path.exists(output_sub_dir):
            os.makedirs(output_sub_dir)
        
        for image_file in tqdm(image_files, desc=f"Processing {os.path.basename(sub_dir)}"):
            preprocess_and_save_png_image(image_file, output_sub_dir, count, new_shape)
            count += 1

    print("Preprocessing completed and saved to output_patches_preprocessed.")

if __name__ == "__main__":
    main()
