#!/usr/bin/env python3

"""
Mesh generation pipeline for cerebral artery reconstruction.
"""
import ants
import os
import sys

# Add the path to make the utils module importable
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
utils_path = os.path.join(project_root, 'analysis_code', 'utils')
preprocessing_path = os.path.join(project_root, 'analysis_code', 'preprocessing')
sys.path.insert(0, os.path.join(project_root, 'analysis_code'))  # Add analysis_code to path for package imports
sys.path.append(preprocessing_path)

# Import function modules from the same directory
import mesh_generation
import preprocessing
import utils
import m1_definition
import centerline_extraction
########################################################
# ===== USER SETTINGS =====
directory_id = "IXI012-HH-1211"

analysis_target = "M1" # "whole" or "M1"
print(f"Analysis target: {analysis_target}")

voxel_size = 0.5  # mm
volume_threshold = 500
print(f"Volume threshold when removing small islands: {voxel_size**3 * volume_threshold} mm^3")
########################################################
temporary_MRA_dir = os.path.join(project_root, 'data', 'preprocessed', 'MRA_preprocessed', 'temporary', directory_id)
utils.ensure_directory_exists(temporary_MRA_dir, verbose=True)

# MCA hysterisis segmentation image (before manual correction)
vessel_hysterisis_path = os.path.join(project_root, 'data', 'preprocessed', 'MRA_preprocessed', 'segmentations', f'{directory_id}_MRA_vessel_hysterisis.nii.gz')
vessel_hysterisis = ants.image_read(vessel_hysterisis_path)

if analysis_target == "whole":
    # MCA segmentation image (after manual correction by 3d slicer)
    mca_path = os.path.join(project_root, 'data', 'preprocessed', 'MRA_preprocessed', 'segmentations', f'{directory_id}_MRA_vessel_segmentation.nii.gz')
    mca = ants.image_read(mca_path)

    # Check the MCA segmentation image header
    mca = preprocessing.match_origin_and_direction(original_img=vessel_hysterisis, target_img=mca)
    mca_arr = mca.numpy()
   

elif analysis_target == "M1":
    # Run m1_definition.py to generate mca_binary_limeninsulae_subtraction.nii.gz
    print("Running M1 definition (limen insulae plane subtraction)...")
    mca_m1_path = m1_definition.m1_definition(directory_id, show_plots=False)
    
    # MCA subtraction image
    mca_m1 = ants.image_read(mca_m1_path)
    
    # Check the MCA subtraction image header
    mca_m1 = preprocessing.match_origin_and_direction(original_img=vessel_hysterisis, target_img=mca_m1)
    mca_arr = mca_m1.numpy()

# Check the MCA subtraction image (TRSAA SyNOnly registered MRA image is registered to the T1-weighted image)
TRSAA_SyNOnly_img_MRA_path = os.path.join(temporary_MRA_dir, f'{directory_id}_MRA_TRSAA_SyNOnly_registered.nii.gz')
TRSAA_SyNOnly_img_MRA = ants.image_read(TRSAA_SyNOnly_img_MRA_path)
TRSAA_SyNOnly_img_MRA_arr = TRSAA_SyNOnly_img_MRA.numpy()

import matplotlib.pyplot as plt
import numpy as np       
plt.figure(figsize=(15, 7.5))
plt.subplot(121)
plt.imshow(np.max(mca_arr, axis=1), alpha=0.5)
plt.subplot(122)
plt.imshow(np.max(TRSAA_SyNOnly_img_MRA_arr, axis=1), alpha=1)
plt.imshow(np.max(mca_arr, axis=1), alpha=0.5)
plt.show()

# Generate initilal vessel mesh with marching cubes
vessel_mc_vtk = mesh_generation.vtk_mc(mesh_generation.np2vtk(mca_arr), threshold=1)
# Calculate volume of the vessel mesh
vol_list = mesh_generation.vol_distribution(vessel_mc_vtk)
# Remove small islands
vessel_mc_vtk = mesh_generation.filter_by_volume(vessel_mc_vtk, volume_threshold)
# Save vessel mesh to temporary directory as ply file
vessel_mc_path = os.path.join(temporary_MRA_dir, f"{directory_id}_vessel_mesh_marching_cubes.ply")
mesh_generation.export_vtkpolydata_to_ply(vessel_mc_vtk, vessel_mc_path)


if analysis_target == "whole":
    # Generate fine vessel mesh with Taubin smoothing and butterfly subdivision
    vessel_mesh_path = os.path.join(project_root, 'data', 'preprocessed', 'MRA_preprocessed', 'vessel_meshes', directory_id, f"{directory_id}_vessel_mesh.ply") 
    utils.ensure_directory_exists(os.path.dirname(vessel_mesh_path), verbose=True)

elif analysis_target == "M1":
    # Generate fine vessel mesh with Taubin smoothing and butterfly subdivision
    temporary_output_dir = os.path.join(project_root, 'data', 'preprocessed', 'MRA_preprocessed', 'temporary', directory_id)
    utils.ensure_directory_exists(temporary_output_dir, verbose=True)

    mca_mesh = mesh_generation.generate_vessel_mesh (vessel_mc_path, 
        output_mesh_path=temporary_output_dir,
        volume_threshold = volume_threshold) # Alreadly saved in the function

    sources_on_mesh_path = os.path.join(project_root, 'data', 'preprocessed', 'MRA_preprocessed', 'vessel_meshes', directory_id, f"{directory_id}_inlets.mrk.json")
    sources_on_mesh = centerline_extraction.load_markups_positions(sources_on_mesh_path)
    
    # Extract M1 object connected to sources_on_mesh
    print("Extracting M1 object connected to sources_on_mesh...")
    mca_mesh = mesh_generation.extract_object_connected_to_points(mca_mesh, sources_on_mesh, verbose=True)

    # Save the extracted M1 mesh (overwrite the existing file at vessel_mesh_path)
    vessel_mesh_path = os.path.join(project_root, 'data', 'preprocessed', 'MRA_preprocessed', 'vessel_meshes', directory_id, f"{directory_id}_{analysis_target}_vessel_mesh.ply")
    mesh_generation.export_vtkpolydata_to_ply(mca_mesh, vessel_mesh_path)

# Check the vessel mesh
import pyvista as pv
pl = pv.Plotter()
pl.add_mesh(mca_mesh, color='lightblue', opacity=0.5)
pl.add_title(f"{analysis_target} vessel mesh", font_size=10)
pl.camera_position = 'xy'
pl.camera.azimuth = 180
pl.camera.zoom(1.2)
pl.show()

print(f"Vessel mesh saved to: {vessel_mesh_path}")