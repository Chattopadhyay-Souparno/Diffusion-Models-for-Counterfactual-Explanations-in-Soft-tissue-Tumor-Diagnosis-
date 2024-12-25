import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt

# Base folder containing the Lipo folders
base_folder = 'Data/worc'
output_base_folder = 'output_slices'

# Ensure the output base folder exists
if not os.path.exists(output_base_folder):
    os.makedirs(output_base_folder)

# Iterate over each Lipo folder
for i in range(1, 116):
    folder_name = f'Lipo-{i:03d}'
    input_folder = os.path.join(base_folder, folder_name)
    
    # Paths to the image and segmentation files
    image_file = os.path.join(input_folder, 'image.nii.gz')
    segmentation_file = os.path.join(input_folder, 'segmentation.nii.gz')
    
    # Check if both the image and segmentation files exist
    if not os.path.exists(image_file) or not os.path.exists(segmentation_file):
        print(f"Files not found: {image_file} or {segmentation_file}")
        continue
    
    # Load the NIFTI files
    img = nib.load(image_file)
    seg = nib.load(segmentation_file)
    
    # Get the data from the NIFTI files
    img_data = img.get_fdata()
    seg_data = seg.get_fdata()
    
    # Get the number of slices
    num_slices = img_data.shape[2]
    
    # Define the output folders for images and masks
    output_image_folder = os.path.join(output_base_folder, 'image', folder_name)
    output_mask_folder = os.path.join(output_base_folder, 'mask', folder_name)
    
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    
    if not os.path.exists(output_mask_folder):
        os.makedirs(output_mask_folder)
    
    # Save each XY slice as a .png file only if it contains foreground information
    for slice_idx in range(num_slices):
        # Select the slice along the z-axis
        img_slice_data = img_data[:, :, slice_idx]
        seg_slice_data = seg_data[:, :, slice_idx]
        
        # Check if the segmentation slice contains white (foreground information)
        if np.any(seg_slice_data > 0):  # This assumes white is any non-zero pixel
            # Save the mask slice
            plt.imshow(seg_slice_data, cmap='gray')
            plt.axis('off')  # Turn off the axis
            mask_output_path = os.path.join(output_mask_folder, f'slice_{slice_idx:03d}.png')
            plt.savefig(mask_output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # Save the corresponding image slice
            plt.imshow(img_slice_data, cmap='gray')
            plt.axis('off')  # Turn off the axis
            image_output_path = os.path.join(output_image_folder, f'slice_{slice_idx:03d}.png')
            plt.savefig(image_output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
    
    print(f"Processed {num_slices} slices for {folder_name}")

print("All slices with foreground information have been saved.")
