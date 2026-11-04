#!/usr/bin/env python3
"""Isotropic remeshing of CAT12 pial surfaces."""
import os
import sys

# Add paths for package imports (same pattern as other mesh runners)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
utils_path = os.path.join(project_root, 'lib')
sys.path.insert(0, os.path.join(project_root, 'lib'))
sys.path.append(utils_path)

import utils
import surface_remesh

########################################################
# ===== USER SETTINGS =====
directory_id = "IXI012-HH-1211"

                                               
remesh_iterations = 10
remesh_targetlen = 0.66   # ~0.33 mm edge length when voxel_size = 0.5 mm
remesh_featuredeg = 30
remesh_maxsurfdist_percent = 1
########################################################

print(f"directory_id: {directory_id}")

# CAT12 outputs (same layout as run_mri_preprocessing.py)
post_cat12_dir = os.path.join(
    project_root, 'data', 'preprocessed', 'T1_preprocessed', 'post_cat12',
    directory_id + "_cat12",
)
post_cat12_mri_dir = os.path.join(post_cat12_dir, 'mri')
post_cat12_surf_dir = os.path.join(post_cat12_dir, 'surf')

preprocessed_img_T1_path = os.path.join(post_cat12_mri_dir, 'm1_1_resampled_T1.nii')
lt_pial_gii_path = os.path.join(post_cat12_surf_dir, 'lh.pial.1_1_resampled_T1.gii')
rt_pial_gii_path = os.path.join(post_cat12_surf_dir, 'rh.pial.1_1_resampled_T1.gii')

required_inputs = {
    'CAT12 mri (m*.nii)': preprocessed_img_T1_path,
    'lh.pial GIFTI': lt_pial_gii_path,
    'rh.pial GIFTI': rt_pial_gii_path,
}
missing = [name for name, path in required_inputs.items() if not os.path.exists(path)]
if missing:
    print("CAT12 surface outputs not found. Please run CAT12 and re-run this script.")
    for name in missing:
        print(f"  Missing: {name} → {required_inputs[name]}")
    sys.exit(0)

# Output directory expected by morphometrics
brain_mesh_dir = os.path.join(
    project_root, 'data', 'preprocessed', 'T1_preprocessed', 'brain_meshes', directory_id
)
utils.ensure_directory_exists(brain_mesh_dir, verbose=True)

remesh_kwargs = dict(
    iterations=remesh_iterations,
    targetlen=remesh_targetlen,
    featuredeg=remesh_featuredeg,
    maxsurfdist_percent=remesh_maxsurfdist_percent,
)

print("\n=== Left hemisphere ===")
lt_pial_path, lt_remesh_path = surface_remesh.process_hemisphere(
    side='lt',
    gii_path=lt_pial_gii_path,
    nii_path=preprocessed_img_T1_path,
    output_dir=brain_mesh_dir,
    directory_id=directory_id,
    remesh_kwargs=remesh_kwargs,
)

print("\n=== Right hemisphere ===")
rt_pial_path, rt_remesh_path = surface_remesh.process_hemisphere(
    side='rt',
    gii_path=rt_pial_gii_path,
    nii_path=preprocessed_img_T1_path,
    output_dir=brain_mesh_dir,
    directory_id=directory_id,
    remesh_kwargs=remesh_kwargs,
)

print("\nDone. Morphometrics-compatible outputs:")
print(f"  {lt_remesh_path}")
print(f"  {rt_remesh_path}")
