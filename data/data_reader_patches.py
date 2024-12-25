import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt


base_folder = 'Data/worc'
output_base_folder = 'output_patches'


if not os.path.exists(output_base_folder):
    os.makedirs(output_base_folder)


for i in range(1, 116):
    folder_name = f'Lipo-{i:03d}'
    input_folder = os.path.join(base_folder, folder_name)
    

    image_file = os.path.join(input_folder, 'image.nii.gz')
    segmentation_file = os.path.join(input_folder, 'segmentation.nii.gz')
    

    if not os.path.exists(image_file) or not os.path.exists(segmentation_file):
        print(f"Files not found: {image_file} or {segmentation_file}")
        continue
    

    img = nib.load(image_file)
    seg = nib.load(segmentation_file)
    

    img_data = img.get_fdata()
    seg_data = seg.get_fdata()
    

    num_slices = img_data.shape[2]
    

    output_image_folder = os.path.join(output_base_folder, 'images', folder_name)
    output_mask_folder = os.path.join(output_base_folder, 'masks', folder_name)
    
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    
    if not os.path.exists(output_mask_folder):
        os.makedirs(output_mask_folder)
    

    for slice_idx in range(num_slices):
        # Select the slice along the z-axis
        img_slice_data = img_data[:, :, slice_idx]
        seg_slice_data = seg_data[:, :, slice_idx]
        
        # Check if the segmentation slice contains white (foreground information)
        if np.any(seg_slice_data > 0):  # This assumes white is any non-zero pixel
            # Get the bounding box of the mask
            rows = np.any(seg_slice_data, axis=1)
            cols = np.any(seg_slice_data, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Crop the image and mask slices
            img_crop = img_slice_data[rmin:rmax+1, cmin:cmax+1]
            seg_crop = seg_slice_data[rmin:rmax+1, cmin:cmax+1]
            
            # Save the cropped mask slice
            plt.imshow(seg_crop, cmap='gray')
            plt.axis('off')  # Turn off the axis
            mask_output_path = os.path.join(output_mask_folder, f'slice_{slice_idx:03d}_mask.png')
            plt.savefig(mask_output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # Save the corresponding cropped image slice
            plt.imshow(img_crop, cmap='gray')
            plt.axis('off')  # Turn off the axis
            image_output_path = os.path.join(output_image_folder, f'slice_{slice_idx:03d}_image.png')
            plt.savefig(image_output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
    
    print(f"Processed {num_slices} slices for {folder_name}")

print("All patches with foreground information have been saved.")
