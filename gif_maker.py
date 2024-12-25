import imageio
import os
import numpy as np

def create_gif(image_folder, output_gif_path, duration=0.1):
    images = []
    file_names = sorted((fn for fn in os.listdir(image_folder) if fn.endswith('.png')))
    for file_name in file_names:
        file_path = os.path.join(image_folder, file_name)
        image = imageio.imread(file_path)
        
        # Check if the image contains white (foreground information)
        if np.any(image > 0):  # This assumes white is any non-zero pixel
            images.append(image)
    
    if images:
        imageio.mimsave(output_gif_path, images, duration=duration)
        print(f"GIF saved at {output_gif_path}")
    else:
        print(f"No images with foreground information found in {image_folder}")

# Folder containing the images
image_folder = 'output_slices/mask/Lipo-001'  # Change this to the appropriate folder
output_gif_path = 'output_slices/Lipo-001_mask.gif'

# Create the GIF
create_gif(image_folder, output_gif_path, duration=0.1)
