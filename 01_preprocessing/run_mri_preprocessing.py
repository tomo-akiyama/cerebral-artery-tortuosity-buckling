#!/usr/bin/env python3
"""MRI preprocessing pipeline: resample, denoise, N4, BET, register, hysteresis segmentation."""

"""
Image preprocessing pipeline for cerebral artery and brain surface reconstruction.

Processing flow:

## 1 T1 1mm * 1mm * 1mm
 ### 1_1  Isotropic resampling 1mm * 1mm * 1mm → 0.5mm * 0.5mm * 0.5mm
 ### 1_2 resampled T1 is sent to CAT12 for automatic segmentation and surface reconstruction

## 2 MRA 0.5mm * 0.5mm * 0.8mm
 ### 2_1 MRA 0.5mm * 0.5mm * 0.8mm → 0.5mm * 0.5mm * 0.5mm
 ### 2_2 Denoising (noise reduction) - ANLM, Gaussian
 ### 2_3 N4 bias field correction - ANTsPy
 ### 2_4 Skull stripping - FSL-BET

## 3 T1 and MRA registration (1_2 is sent to CAT12 for automatic segmentation and surface reconstruction, and then preprocessed T1 by CAT12 is registered to the preprocessed MRA)
 ### 3_1 Linear registration (TRSAA)
 ### 3_2 Nonlinear registration (SyNOnly)

## 4 Hysteresis vessel segmentation
 ### 4_1 Hysteresis vessel segmentation from the preprocessed MRA
"""

import ants
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from brainextractor import BrainExtractor
from skimage.morphology import remove_small_objects
import os
import sys

# Add the path to make the utils module importable
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
utils_path = os.path.join(project_root, 'lib')
sys.path.insert(0, os.path.join(project_root, 'lib'))  # Add analysis_code to path for package imports
sys.path.append(utils_path)

# Import function modules from the same directory
import preprocessing

# Import from utils
import utils
import tool


########################################################
# ===== USER SETTINGS =====
directory_id = "IXI013-HH-1212"

voxel_size = 0.5  # mm
volume_threshold = 5000; print(f"Volume threshold when removing small islands: {volume_threshold} voxels, {voxel_size**3 * volume_threshold} mm^3")
########################################################

case_id = directory_id[3:6].lstrip("0")
case_id = int(case_id)
print("directory_id", directory_id)
print("case_id", case_id)

# Load the images
data_raw_dir = os.path.join(project_root, 'data', 'raw')
input_T1_path = os.path.join(data_raw_dir, 'T1_HH', f'{directory_id}-T1.nii.gz')
input_MRA_path = os.path.join(data_raw_dir, 'MRA_HH', f'{directory_id}-MRA.nii.gz')

# Ensure files are available (download from Dropbox if online-only)
print("Checking T1 image availability...")
if not utils.ensure_file_available(input_T1_path, max_wait_time=120):
    raise FileNotFoundError(f"T1 image is not available: {input_T1_path}")

print("Checking MRA image availability...")
if not utils.ensure_file_available(input_MRA_path, max_wait_time=120):
    raise FileNotFoundError(f"MRA image is not available: {input_MRA_path}")

# Load the images
img_T1 = ants.image_read(input_T1_path)
img_MRA = ants.image_read(input_MRA_path)


########################################################################################################################################################
#### 1 Preprocessing for T1 ####
########################################################################################################################################################
# 1_1  Isotropic resampling 1mm * 1mm * 1mm → 0.5mm * 0.5mm * 0.5mm
print("T1 original image header", img_T1)
# Isotropic resampling for T1
print("Resampling T1 image...")
resampled_img_T1 = ants.resample_image(img_T1, (voxel_size, voxel_size, voxel_size),False,4)
print("T1 image header after resampling", resampled_img_T1)

# Set negative voxel values in the background to 0 (may occur due to resampling artifacts)
resampled_img_T1[resampled_img_T1.numpy() < 0] = 0

# Save the resampled T1 image
resampled_T1_dir = os.path.join(project_root, 'data', 'preprocessed', 'T1_preprocessed', 'resampled', directory_id)
utils.ensure_directory_exists(resampled_T1_dir, verbose=True) # Create the directory if it does not exist
resampled_img_T1_path = os.path.join(resampled_T1_dir, f'{directory_id}_T1_resampled.nii.gz')
ants.image_write(resampled_img_T1, resampled_img_T1_path, ri=False) # 1_2: This image is sent to CAT12 for automatic segmentation and surface reconstruction
########################################################################################################################################################


########################################################################################################################################################
#### 2 Preprocessing for MRA ####
########################################################################################################################################################
# 2_1 MRA 0.5mm * 0.5mm * 0.8mm → 0.5mm * 0.5mm * 0.5mm
print("MRA original image header", img_MRA)
# Isotropic resampling for MRA
print("Resampling MRA image...")
resampled_img_MRA = ants.resample_image(img_MRA, (voxel_size, voxel_size, voxel_size),False,4)

# Add pad
print("Adding padding to MRA image...")
matrix_size = max(resampled_img_MRA.shape)
resampled_img_MRA = ants.pad_image(resampled_img_MRA, shape=(matrix_size, matrix_size, matrix_size)) 
print("MRA image header after resampling and padding", resampled_img_MRA)

# Set negative voxel values in the background to 0 (may occur due to resampling artifacts)
resampled_img_MRA[resampled_img_MRA.numpy() < 0] = 0

# Save the resampled MRA image (temporary directory for segmentation)
temporary_MRA_dir = os.path.join(project_root, 'data', 'preprocessed', 'MRA_preprocessed', 'temporary', directory_id)
utils.ensure_directory_exists(temporary_MRA_dir, verbose=True) # Create the directory if it does not exist
resampled_img_MRA_path = os.path.join(temporary_MRA_dir, f'{directory_id}_MRA_resampled.nii.gz')
ants.image_write(resampled_img_MRA, resampled_img_MRA_path, ri=False)

# 2_2 Denoising (noise reduction) - ANLM, Gaussian
# Adaptive Non-Local Means Denoising with Gaussian noise model
print("Denoising MRA image...")
denoised_img_MRA = ants.denoise_image(resampled_img_MRA, 
                                    mask=None, 
                                    shrink_factor=1, 
                                    p=1, r=3, 
                                    noise_model='Gaussian', v=0)
print("Denoised MRA image header", denoised_img_MRA)
# Save the denoised MRA image (temporary directory for segmentation)
denoised_img_MRA_path = os.path.join(temporary_MRA_dir, f'{directory_id}_MRA_denoised.nii.gz')
ants.image_write(denoised_img_MRA, denoised_img_MRA_path, ri=False)

# 2_3 N4 bias field correction - ANTsPy
print("N4 bias field correction for MRA image...")
bias_corrected_img_MRA = ants.n4_bias_field_correction(denoised_img_MRA, 
                                                  rescale_intensities=True, 
                                                  shrink_factor=2)
print("Bias corrected MRA image header", bias_corrected_img_MRA)

# Save the corrected MRA image (temporary directory for segmentation)
bias_corrected_img_MRA_path = os.path.join(temporary_MRA_dir, f'{directory_id}_MRA_bias_corrected.nii.gz')
ants.image_write(bias_corrected_img_MRA, bias_corrected_img_MRA_path, ri=False)

# 2_4 Skull stripping - FSL-BET
print("Skull stripping for MRA image...")
preprocessed_img_MRA = nib.load(bias_corrected_img_MRA_path) # Load the bias corrected MRA image as a nibabel image
bet_mask = BrainExtractor(img=preprocessed_img_MRA, bt=0.45)                                                                                             
bet_mask.run()

# Save the bet mask image (temporary directory for segmentation)
bet_mask_MRA_path = os.path.join(temporary_MRA_dir, f'{directory_id}_MRA_bet_mask.nii.gz')
bet_mask.save_mask(bet_mask_MRA_path)

# Brain extraction modifies the image origin; restore the original origin to maintain spatial consistency
bet_mask_MRA = ants.image_read(bet_mask_MRA_path)
print("Bet mask MRA image header before matching origin and direction", bet_mask_MRA)
# Check if the origin and direction are matched, and if not, update the origin and direction to match the previous image
bet_mask_MRA = preprocessing.match_origin_and_direction(original_img=bias_corrected_img_MRA, target_img=bet_mask_MRA) 
ants.image_write(bet_mask_MRA, bet_mask_MRA_path, ri=False)
print("Bet mask MRA image header", bet_mask_MRA)

bet_mask_MRA_arr = bet_mask_MRA.numpy()
brain_MRA = bias_corrected_img_MRA * bet_mask_MRA_arr
# Save the brain MRA image (temporary directory for segmentation)   
brain_MRA_path = os.path.join(temporary_MRA_dir, f'{directory_id}_MRA_brain.nii.gz')
ants.image_write(brain_MRA, brain_MRA_path, ri=False)
print("Brain MRA image header", brain_MRA)
########################################################################################################################################################


########################################################################################################################################################
#### 3 T1 and MRA registration ####
########################################################################################################################################################
# 3 T1 and MRA registration (1_2 is sent to CAT12 for automatic segmentation and surface reconstruction, and then preprocessed T1 by CAT12 is registered to the preprocessed MRA)
# 3_1 Linear registration (TRSAA)
moving_mask = bet_mask_MRA 

# CAT12 outputs: m* = fixed image for registration; p* = source of fixed_mask
post_cat12_T1_dir = os.path.join(project_root, 'data', 'preprocessed', 'T1_preprocessed', 'post_cat12', directory_id+"_cat12", "mri")
preprocessed_img_T1_path = os.path.join(post_cat12_T1_dir, 'm1_1_resampled_T1.nii')
segmented_brain_T1_path = os.path.join(post_cat12_T1_dir, 'p01_1_resampled_T1.nii')
if not (os.path.exists(preprocessed_img_T1_path) and os.path.exists(segmented_brain_T1_path)): # Check if the CAT12 output files exist
    print("CAT12 output not found. Please run CAT12 and re-run this script.")
    sys.exit(0)
"""
CAT12 output files (see: https://andysbrainbook.readthedocs.io/en/latest/CAT12/CAT12_03_Preprocessing.html, and https://neuro-jena.github.io/cat12-help/?utm_source=chatgpt.com)  
- mwp1: modulated rey matter segmentation) (mwp1_* files)
- mwp2: modulated white matter segmentation (mwp2_* files)
- p: segmented anatomical image with the skull and face removed (p_* files)
- wm: segmented anatomical scan warped to MNI space (wm_* files)
- y: inverse deformation field, used to identify and correct for field inhomogeneities (y_* files)
- m: bias-, noise-, and intensity-corrected T1 image in native space
"""

# Load CAT12 processed T1 image (m*: preprocessed anatomical image in native space)
preprocessed_img_T1 = ants.image_read(preprocessed_img_T1_path)

# Use the CAT12 "p" image (skull- and face-stripped) only to build the fixed mask
segmented_brain_T1 = ants.image_read(segmented_brain_T1_path)
fixed_mask = ants.get_mask(segmented_brain_T1)

# Set the number of threads to 1 to ensure complete reproducibility
import os
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

                                                                                
print("Initial registration with TRSAA...")
TRSAA_registration = ants.registration(
    fixed=preprocessed_img_T1, 
    moving=bias_corrected_img_MRA, 
    type_of_transform='TRSAA',
    mask=fixed_mask, 
    moving_mask=moving_mask, 
    aff_metric='mattes',                                                                                                                                                         
    aff_random_sampling_rate=0.2,
    aff_sampling=256,
    aff_iterations=(3000, 2000, 1500, 500), 
    aff_shrink_factors=(8, 4, 2, 1),
    reg_iterations = (1000,1000,100,20), 
    random_seed=12345, 
    verbose=False)

# TRSAA registered MRA image
TRSAA_moving_img_MRA = TRSAA_registration['warpedmovout']
# Save the TRSAA registered MRA image (temporary directory for segmentation)
TRSAA_moving_img_MRA_path = os.path.join(temporary_MRA_dir, f'{directory_id}_MRA_TRSAA_registered.nii.gz')
ants.image_write(TRSAA_moving_img_MRA, TRSAA_moving_img_MRA_path, ri=False)
print("TRSAA registered MRA image header", TRSAA_moving_img_MRA)

### 3_2 Nonlinear registration (SyNOnly)
# Apply SyNOnly registration following TRSAA-based initial alignment
# Use the TRSAA transform as the initial_transform for the subsequent nonlinear registration
print("Nonlinear registration with SyNOnly...")
TRSAA_SyNOnly_registration = ants.registration(
    fixed=preprocessed_img_T1, 
    moving=bias_corrected_img_MRA, 
    type_of_transform='SyNOnly',
    mask=fixed_mask, 
    moving_mask=moving_mask, 
    initial_transform=TRSAA_registration['fwdtransforms'],
    syn_metric='mattes',                                                     
    syn_sampling=256, 
    reg_iterations = (1000,1000,100,20),
    random_seed=12345,  # Ensure reproducibility by setting a random seed
    verbose=False)

# TRSAA SyNOnly registered MRA image
TRSAA_SyNOnly_moving_img_MRA = TRSAA_SyNOnly_registration['warpedmovout']
# Save the TRSAA SyNOnly registered MRA image (temporary directory for segmentation)
TRSAA_SyNOnly_moving_img_MRA_path = os.path.join(temporary_MRA_dir, f'{directory_id}_MRA_TRSAA_SyNOnly_registered.nii.gz')
ants.image_write(TRSAA_SyNOnly_moving_img_MRA, TRSAA_SyNOnly_moving_img_MRA_path)
print("TRSAA and SyNOnly registered MRA image header after TRSAA registration", TRSAA_SyNOnly_moving_img_MRA)

# Use the SyNOnly-registered brain mask to restrict the MRA to the brain region
mapped_TRSAA_SyNOnly_bet_mask_MRA = ants.apply_transforms(
    fixed=preprocessed_img_T1, 
    moving=moving_mask, 
    transformlist=TRSAA_SyNOnly_registration['fwdtransforms'], 
    interpolator='nearestNeighbor', 
    verbose=False)
# Save the mapped SyNOnly-registered brain mask image (temporary directory for segmentation)
mapped_TRSAA_SyNOnly_bet_mask_MRA_path = os.path.join(temporary_MRA_dir, f'{directory_id}_MRA_TRSAA_SyNOnly_mapped_bet_mask.nii.gz')
ants.image_write(mapped_TRSAA_SyNOnly_bet_mask_MRA, mapped_TRSAA_SyNOnly_bet_mask_MRA_path, ri=False)

# Restrict the MRA to the brain region
brain_MRA = TRSAA_SyNOnly_moving_img_MRA * mapped_TRSAA_SyNOnly_bet_mask_MRA
# Convert the brain MRA image to a numpy array
brain_MRA_arr = brain_MRA.numpy()
print("Brain MRA image header after TRSAA and SyNOnly registration", brain_MRA)

########################################################################################################################################################
#### 4 Hysteresis vessel segmentation ####
########################################################################################################################################################
# 4_1 Hysteresis vessel segmentation from the preprocessed MRA
# Calculate the signals of the brain MRA image
brain_MRA_signals = brain_MRA_arr.flatten()
brain_MRA_signals = brain_MRA_signals[brain_MRA_signals > 0]

# Define the thresholds for the hysteresis vessel segmentation
high_threshold_percentile = 99.9
low_threshold_percentile = 98.5
low_threshold = np.percentile(brain_MRA_signals, low_threshold_percentile)
high_threshold = np.percentile(brain_MRA_signals, high_threshold_percentile)

# Plot the histogram of the brain MRA signals
plt.figure()
plt.hist(brain_MRA_signals, bins=1000, alpha=0.5, label="bet_mra", density=True)
plt.axvline(low_threshold, color="red")
plt.axvline(high_threshold, color="green")
plt.text(low_threshold, 0.001, f"MIP with low threshold: {low_threshold:.2f}", color="red")
plt.text(high_threshold, 0.001, f"MIP with high threshold: {high_threshold:.2f}", color="green")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(np.max(brain_MRA_arr>low_threshold, axis=1))
plt.title("low_threshold")
plt.subplot(122)
plt.imshow(np.max(brain_MRA_arr>high_threshold, axis=1))
plt.title("high_threshold")
plt.show()

# Apply hysteresis thresholding to the brain MRA image (Note that fill holes is included in the hysteresis thresholding process)
print("Applying hysteresis thresholding to the brain MRA image...")
vessel_hysterisis_thresholding = preprocessing.apply_hysteresis_threshold(image=brain_MRA_arr, 
                                                     low=low_threshold, 
                                                     high=high_threshold)

# Plot the MIP of the vessel hysteresis thresholding image
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(np.max(vessel_hysterisis_thresholding , axis=2))
plt.subplot(132)
plt.imshow(np.max(vessel_hysterisis_thresholding , axis=1))
plt.subplot(133)
plt.imshow(np.max(vessel_hysterisis_thresholding , axis=0))
plt.suptitle("Vessel hysterisis thresholding (MIP)")
plt.show()

# Remove small islands from the vessel hysteresis thresholding image
vessel_hysterisis_thresholding= remove_small_objects(vessel_hysterisis_thresholding.astype(bool), min_size=volume_threshold, connectivity=3).astype('float32')

# Plot the MIP of the vessel hysteresis thresholding image after removing small islands
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(np.max(vessel_hysterisis_thresholding , axis=2))
plt.subplot(132)
plt.imshow(np.max(vessel_hysterisis_thresholding , axis=1))
plt.subplot(133)
plt.imshow(np.max(vessel_hysterisis_thresholding , axis=0))
plt.suptitle("Vessel hysterisis thresholding (MIP) after removing small islands")
plt.show()


# Save the vessel hysteresis thresholding image
segmented_MRA_dir = os.path.join(project_root, 'data', 'preprocessed', 'MRA_preprocessed', 'segmentations', directory_id)
utils.ensure_directory_exists(segmented_MRA_dir, verbose=True) # Create the directory if it does not exist
vessel_segmentation_path = os.path.join(segmented_MRA_dir, f'{directory_id}_MRA_vessel_hysterisis.nii.gz')
vessel_seg_origin = TRSAA_SyNOnly_moving_img_MRA.origin
vessel_seg_spacing = TRSAA_SyNOnly_moving_img_MRA.spacing
vessel_seg_direction = TRSAA_SyNOnly_moving_img_MRA.direction
vessel = ants.from_numpy(vessel_hysterisis_thresholding)
vessel.set_origin(vessel_seg_origin)
vessel.set_spacing(vessel_seg_spacing)
vessel.set_direction(vessel_seg_direction)
print("MRA vessel segmentation image header", vessel)
ants.image_write(vessel, vessel_segmentation_path, ri=False)

