#!/usr/bin/env python3
"""VMTK centerline extraction, smoothing, and arc-length resampling."""

"""
Centerline extraction pipeline for cerebral artery reconstruction.
"""
import ants
import os
import sys
import json
import numpy as np
import pyvista as pv


# Add the path to make the utils module importable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
utils_path = os.path.join(project_root, 'lib')
sys.path.insert(0, os.path.join(project_root, 'lib'))  # Add analysis_code to path for package imports
sys.path.append(utils_path)

# Import function modules from the same directory
import mesh_generation
from lib import file_utils as utils
import centerline_extraction

########################################################
# ===== USER SETTINGS =====
directory_id = "IXI516-HH-2297"

analysis_target = "whole" # "whole" or "M1"
voxel_size = 0.5  # mm

# Reference: 2021Biomed Eng Online Kjeldsberg (https://doi.org/10.1186/s12938-021-00957-6)
smooth_factor = 1.2
smoothing_iterations = 100
resampling_step_length = 0.2; print(f"step length: {resampling_step_length * voxel_size} mm")
########################################################

vessel_mesh_dir = os.path.join(project_root, 'data', 'preprocessed', 'MRA_preprocessed', 'vessel_meshes', directory_id)
output_centerline_dir = os.path.join(project_root, 'data', 'derived', 'vessel_centerlines', directory_id)
utils.ensure_directory_exists(output_centerline_dir, verbose=True)

temporary_derived_centerlines_dir = os.path.join(project_root, 'data', 'derived', 'temporary', 'centerlines', directory_id)
utils.ensure_directory_exists(temporary_derived_centerlines_dir, verbose=True)

# Load vessel mesh
vessel_mesh_path = os.path.join(vessel_mesh_dir, f"{directory_id}_vessel_mesh.ply")

# Load sources and targets from JSON files
sources_on_mesh_path = os.path.join(vessel_mesh_dir, f"{directory_id}_inlets.mrk.json")
targets_on_mesh_path = os.path.join(vessel_mesh_dir, f"{directory_id}_outlets.mrk.json")

sources_on_mesh = centerline_extraction.load_markups_positions(sources_on_mesh_path)
targets_on_mesh = centerline_extraction.load_markups_positions(targets_on_mesh_path)

# Extract centerlines
print("Extracting centerlines...")
centerlines = centerline_extraction.vmtk_extract_centerline(vessel_mesh_path, sources_on_mesh, targets_on_mesh);

if centerlines.GetNumberOfPoints() == 0:
    raise ValueError("Centerline extraction failed: extracted centerlines is empty. Check sources and targets.")


# Splitting centerlines into multiple segments
print("Splitting centerlines into multiple segments...")
centerlines = centerline_extraction.vmtk_branch_splitting(centerlines)

# Smoothing centerlines by Laplacian smoothing filter
print("Smoothing centerlines...")
smoothed_centerlines = centerline_extraction.vmtk_centerline_smoothing(centerlines, num_iteration=smoothing_iterations, smooth_factor=smooth_factor)

if smoothed_centerlines.GetNumberOfPoints() == 0:
    raise ValueError("Centerline smoothing failed: smoothed centerlines is empty.")

# Merging centerlines belonging to the same groups with resampling process
print("Merging centerlines...")
merged_centerlines = centerline_extraction.vmtk_centerlinemerge(smoothed_centerlines, step_length=resampling_step_length)
# Assign arc-length parameterization and normal vectors
merged_centerlines = centerline_extraction.vmtk_centerline_attributes(merged_centerlines)
# Compute geometric quantities
merged_centerlines = centerline_extraction.vmtk_compute_geometry(merged_centerlines)

# Visualize merged centerlines for visual inspection
vessel_mesh = mesh_generation.load_ply_to_polydata(vessel_mesh_path)
pl = pv.Plotter()
pl.add_mesh(merged_centerlines, color='lime', opacity=1, line_width=10)
pl.add_mesh(vessel_mesh, color='lightblue', opacity=0.5)
pl.add_points(sources_on_mesh, color="magenta", opacity=0.8, point_size=15, render_points_as_spheres=True,ambient=1.0)
pl.add_points(targets_on_mesh, color="cyan", opacity=0.8, point_size=15, render_points_as_spheres=True,ambient=1.0)
pl.add_title("Centerlines extracted from vessel mesh by VMTK", font_size=10)
pl.camera_position = 'xy'
pl.camera.azimuth = 180
pl.camera.zoom(1.2)

               
def close_on_space():
    pl.close()

pl.add_key_event('space', close_on_space)
pl.add_text("Press SPACE to close and save", position='lower_left', font_size=14, color='red')
print("Showing visualization window... (press SPACE to close and save)")

try:
    pl.show(interactive=True)
except (AttributeError, RuntimeError):
                                                            
    pass

# Save centerlines after visual inspection
print("Saving centerlines...")
centerlines_path = os.path.join(temporary_derived_centerlines_dir, f"{directory_id}_centerlines.vtp")
centerline_extraction.save_vtp(centerlines, centerlines_path)

# Save smoothed centerlines after visual inspection
print("Saving smoothed centerlines...")
smoothed_centerlines_path = os.path.join(temporary_derived_centerlines_dir, f"{directory_id}_smoothed_centerlines.vtp")
centerline_extraction.save_vtp(smoothed_centerlines, smoothed_centerlines_path)

# Save merged centerlines after visual inspection
print("Saving merged centerlines...")
merged_centerlines_path = os.path.join(output_centerline_dir, f"{directory_id}_merged_centerlines.vtp")
centerline_extraction.save_vtp(merged_centerlines, merged_centerlines_path)

print(f"Centerlines saved to: {output_centerline_dir}")