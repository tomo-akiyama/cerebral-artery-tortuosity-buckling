#!/usr/bin/env python3
"""Map limen insulae reference planes and define distal M1 endpoints."""

"""
Transform limen insulae reference points defined in MNI space to native space, 
and create subtraction image of MCA image and limen insulae plane in native space. 
"""

import ants
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add the path to make the preprocessing module importable
project_root_for_import = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
preprocessing_path = os.path.join(project_root_for_import, '01_preprocessing')
sys.path.insert(0, os.path.join(project_root_for_import, 'lib'))
sys.path.append(preprocessing_path)
import preprocessing

########################################################
# ===== USER SETTINGS =====

# Limen insulae reference points in voxel space <--> world space (mm)
rt_voxel_coords = np.array([131, 140, 60]) # voxel space <--> world space (mm) [35, 8, -18]
lt_voxel_coords = np.array([61, 140, 60])  # voxel space <--> world space (mm) [-35, 8, -18]
########################################################

def m1_definition(directory_id, show_plots=True):
    """
    Transform limen insulae reference points defined in MNI space to native space.
    
    Args:
        directory_id: Directory ID for the case
        show_plots: Whether to show plots (default: True)
    
    Returns:
        Path to the generated subtraction image
    """
    # Load the MNI template image
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mni_template_path = os.path.join(project_root, 'data', 'raw', 'CAT12 template', 'Template_T1.nii')
    img_mni = ants.image_read(mni_template_path)

    # Load the T1 image
    """
    CAT12 output files (see: https://andysbrainbook.readthedocs.io/en/latest/CAT12/CAT12_03_Preprocessing.html)  
    - mwp1: Modulated gray matter segmentation (mwp1_* files)
    - mwp2: Modulated white matter segmentation (mwp2_* files)
    - p: Skull- and face-stripped anatomical image (p_* files)
    - wm: Anatomical image warped to MNI space (wm_* files)
    - y: Inverse deformation field for bias field correction (y_* files)
    - mi: Preprocessed anatomical image in native space
    """
    post_cat12_T1_dir = os.path.join(project_root, 'data', 'preprocessed', 'T1_preprocessed', 'post_cat12', directory_id+"_cat12", "mri")
    preprocessed_img_T1_path = os.path.join(post_cat12_T1_dir, 'm1_1_resampled_T1.nii')
    img_t1 = ants.image_read(preprocessed_img_T1_path)

    # Registration of T1 image to MNI template image
    registration_result = ants.registration(fixed=img_mni, moving=img_t1, type_of_transform='SyN')
    # forward transform and inverse transform
    forward_transform = registration_result['fwdtransforms']
    inverse_transform = registration_result['invtransforms']
    # Transform the T1 image to MNI space
    warped_img_t1 = ants.apply_transforms(fixed=img_mni, moving=img_t1, transformlist=forward_transform, interpolation='linear')

    # Check the transformation results in the image
    z = lt_voxel_coords[2]
    lt_y_plane = lt_voxel_coords[0]
    rt_y_plane = rt_voxel_coords[0]

    if show_plots:
        plt.figure(figsize=(15, 7.5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_mni.numpy()[:, :, z], cmap='gray')
        plt.plot(rt_voxel_coords[1], rt_voxel_coords[0], 'ro', label='Rt')
        plt.plot(lt_voxel_coords[1], lt_voxel_coords[0], 'yo', label='Lt')
        plt.axhline(y=rt_y_plane, color='r', linestyle='--', label='Rt plane')
        plt.axhline(y=lt_y_plane, color='y', linestyle='--', label='Lt plane')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("MNI space")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.imshow(warped_img_t1.numpy()[:, :, z], cmap='gray')
        plt.plot(rt_voxel_coords[1], rt_voxel_coords[0], 'ro', label='Rt')
        plt.plot(lt_voxel_coords[1], lt_voxel_coords[0], 'yo', label='Lt')
        plt.axhline(y=rt_y_plane, color='r', linestyle='--', label='Rt plane')
        plt.axhline(y=lt_y_plane, color='y', linestyle='--', label='Lt plane')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("T1-weighted image in MNI space")
        plt.legend()
        plt.show()

    # Create limen insulae plane in MNI space
    shape_warped = warped_img_t1.numpy().shape
    lt_limeninsulae_plane_array = np.zeros(shape_warped, dtype=np.float32)
    lt_limeninsulae_plane_array[lt_y_plane, :, :] = 1.0 # order: y, x, z
    rt_limeninsulae_plane_array = np.zeros(shape_warped, dtype=np.float32)
    rt_limeninsulae_plane_array[rt_y_plane, :, :] = 1.0 # order: y, x, z

    lt_limeninsulae_plane = ants.from_numpy(lt_limeninsulae_plane_array, 
                                            origin=img_mni.origin, 
                                            spacing=img_mni.spacing, 
                                            direction=img_mni.direction)
    rt_limeninsulae_plane = ants.from_numpy(rt_limeninsulae_plane_array,
                                            origin=img_mni.origin, 
                                            spacing=img_mni.spacing, 
                                            direction=img_mni.direction)
    # Transform the limen insulae plane in MNI space to native space using the inverse transform
    lt_native_plane = ants.apply_transforms(fixed=img_t1, moving=lt_limeninsulae_plane,
                                            transformlist=inverse_transform, interpolator='nearestNeighbor')
    rt_native_plane = ants.apply_transforms(fixed=img_t1, moving=rt_limeninsulae_plane,
                                            transformlist=inverse_transform, interpolator='nearestNeighbor')

    limen_native_plane = lt_native_plane + rt_native_plane

    # Create limen insulae reference points in native space
    lt_limeninsulae_reference_array = np.zeros(shape_warped, dtype=np.float32)
    lt_limeninsulae_reference_array[lt_voxel_coords[0], lt_voxel_coords[1], lt_voxel_coords[2]] = 1 # order: y, x, z
    rt_limeninsulae_reference_array = np.zeros(shape_warped, dtype=np.float32)
    rt_limeninsulae_reference_array[rt_voxel_coords[0], rt_voxel_coords[1], rt_voxel_coords[2]] = 1 # order: y, x, z
    lt_limeninsulae_reference = ants.from_numpy(lt_limeninsulae_reference_array,
                                                origin=img_mni.origin,
                                                spacing=img_mni.spacing,
                                                direction=img_mni.direction)
    rt_limeninsulae_reference = ants.from_numpy(rt_limeninsulae_reference_array,
                                                    origin=img_mni.origin,
                                                    spacing=img_mni.spacing,
                                                    direction=img_mni.direction)
    lt_native_reference = ants.apply_transforms(fixed=img_t1, moving=lt_limeninsulae_reference,
                                                transformlist=inverse_transform, interpolator='nearestNeighbor')
    rt_native_reference = ants.apply_transforms(fixed=img_t1, moving=rt_limeninsulae_reference,
                                            transformlist=inverse_transform, interpolator='nearestNeighbor')

    # Check the transformation results in the image
    # Select the slice where the reference points coincide
    lt_indices = np.where(lt_native_reference.numpy() == 1)[1]
    rt_indices = np.where(rt_native_reference.numpy() == 1)[1]
    common_indices = np.intersect1d(lt_indices, rt_indices)
    if common_indices.size == 0:
        print("No common indices found.")
        slice_index = lt_indices[0] # Select the index of Lt
        print("Selected index:", slice_index)
    else:
        print("Common indices along axis 1:", common_indices)
        slice_index = common_indices[0] # Select the first common index

    if show_plots:
        plt.figure(figsize=(12,6))
        plt.imshow(img_t1.numpy()[:, slice_index, :].T, cmap='gray', origin='lower')
        plt.contour(limen_native_plane.numpy()[:, slice_index, :].T, colors='w', linewidths=0.5)
        plt.title("Bilateral limen insulae plane in native space")
        plt.show()

    # Save the results
    temporary_T1_dir = os.path.join(project_root, 'data', 'preprocessed', 'T1_preprocessed', 'temporary', directory_id)
    os.makedirs(temporary_T1_dir, exist_ok=True)
    ants.image_write(lt_native_plane, os.path.join(temporary_T1_dir, "lt_limeninsulae_plane.nii"))
    ants.image_write(rt_native_plane, os.path.join(temporary_T1_dir, "rt_limeninsulae_plane.nii"))
    ants.image_write(limen_native_plane, os.path.join(temporary_T1_dir, "bil_limeninsulae_plane.nii"))
    ants.image_write(lt_native_reference, os.path.join(temporary_T1_dir, "lt_limeninsulae_reference.nii"))
    ants.image_write(rt_native_reference, os.path.join(temporary_T1_dir, "rt_limeninsulae_reference.nii"))

    # Create subtraction image
    mca_path = os.path.join(project_root, 'data', 'preprocessed', 'MRA_preprocessed', 'segmentations', f'{directory_id}_MRA_vessel_segmentation.nii.gz')
    mca = ants.image_read(mca_path)
    
    # Match origin and direction of limen_native_plane to mca before subtraction
    limen_native_plane = preprocessing.match_origin_and_direction(original_img=mca, target_img=limen_native_plane)
    
    subtraction = mca - limen_native_plane
    subtraction[subtraction == -1.0] = 0.0
    
    # Save subtraction image to MRA temporary directory (for use in run_mesh_generation.py)
    temporary_MRA_dir = os.path.join(project_root, 'data', 'preprocessed', 'MRA_preprocessed', 'temporary', directory_id)
    os.makedirs(temporary_MRA_dir, exist_ok=True)
    mca_m1_path = os.path.join(temporary_MRA_dir, "mca_limeninsulae_subtraction.nii.gz")
    ants.image_write(subtraction, mca_m1_path)
    
    return mca_m1_path

if __name__ == "__main__":
    # Default settings for direct execution
    directory_id = "IXI313-HH-2241"
    m1_definition(directory_id, show_plots=True)