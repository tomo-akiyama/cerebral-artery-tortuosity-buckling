#!/usr/bin/env python3
"""Extract M1 centerline length, Euclidean distance, and tortuosity index."""

"""
Extract features from the M1 mesh.
"""

import os
import sys
import numpy as np
import pyvista as pv
import pandas as pd

# Add the path to make the utils module importable
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
utils_path = os.path.join(project_root, 'lib')
mesh_path = os.path.join(project_root, '03_mesh')
sys.path.insert(0, os.path.join(project_root, 'lib'))
sys.path.append(mesh_path)

import utils
import mesh_generation
import centerline_extraction
import vessel_mesh_features


########################################################
# ===== USER SETTINGS =====
directory_id = "IXI012-HH-1211"

analysis_target = "M1" # fixed
voxel_size = 0.5 # mm
########################################################

# Load vessel mesh
vessel_mesh_dir = os.path.join(project_root, 'data', 'preprocessed', 'MRA_preprocessed', 'vessel_meshes', directory_id)
m1_mesh_path = os.path.join(vessel_mesh_dir, f"{directory_id}_{analysis_target}_vessel_mesh.ply")
m1_mesh = mesh_generation.load_ply_to_polydata(m1_mesh_path)

# Load sources and targets from JSON files
sources_on_mesh_path = os.path.join(vessel_mesh_dir, f"{directory_id}_{analysis_target}_inlets.mrk.json")
targets_on_mesh_path = os.path.join(vessel_mesh_dir, f"{directory_id}_{analysis_target}_outlets.mrk.json")

sources_on_mesh = centerline_extraction.load_markups_positions(sources_on_mesh_path)
targets_on_mesh = centerline_extraction.load_markups_positions(targets_on_mesh_path)

# Extract centerlines
print("Extracting M1 centerlines...")
centerlines = centerline_extraction.vmtk_extract_centerline(m1_mesh_path, sources_on_mesh, targets_on_mesh);
# Compute centerline attributes and geometric features
centerlines = centerline_extraction.vmtk_centerline_attributes(centerlines)
centerlines = centerline_extraction.vmtk_compute_geometry(centerlines)

centerlines_path = os.path.join(project_root, 'data', 'derived', 'vessel_centerlines', directory_id, f"{directory_id}_M1_centerlines.vtp")
utils.ensure_directory_exists(os.path.dirname(centerlines_path), verbose=True)
mesh_generation.save_vtp(centerlines, centerlines_path)




# Check the centerlines
vessel_mesh = mesh_generation.load_ply_to_polydata(os.path.join(vessel_mesh_dir, f"{directory_id}_vessel_mesh.ply"))
pl = pv.Plotter()
pl.add_mesh(centerlines, color='lime', opacity=1, line_width=5)
pl.add_mesh(m1_mesh, color='lightblue', opacity=0.8)
pl.add_mesh(vessel_mesh, color='lightblue', opacity=0.2)
pl.add_points(sources_on_mesh, color="magenta", opacity=0.8, point_size=15, render_points_as_spheres=True,ambient=1.0)
pl.add_points(targets_on_mesh, color="yellow", opacity=0.8, point_size=15, render_points_as_spheres=True,ambient=1.0)
pl.add_title("M1 centerlines and vessel mesh")
pl.camera_position = 'xy'
pl.camera.azimuth = 180
pl.camera.zoom(1.2)
pl.show() 



# Extract features from the M1 mesh
celldata, pointids, points, pointdata = centerline_extraction.vmtk_convert2numpy(centerlines)

                                                 
right_cells, left_cells, threshold_x, is_right_point, is_left_point = vessel_mesh_features.classify_centerline_left_right_by_kmeans(
    points, pointids, n_iter=100, verbose=True
)
# Split mesh left/right
right_vessel, left_vessel, right_center_x, left_center_x = vessel_mesh_features.split_mesh_left_right(
    m1_mesh, threshold_x, verbose=True
    )

pl = pv.Plotter()
pl.add_mesh(vessel_mesh, color='lightblue', opacity=0.2)
pl.add_mesh(right_vessel, color='green', opacity=0.8)
pl.add_mesh(left_vessel, color='blue', opacity=0.8)
pl.add_title("Right (green) and left (blue)")
pl.camera_position = 'xy'
pl.camera.azimuth = 180
pl.camera.zoom(1.2)
pl.show() 

endpoints = np.concatenate([sources_on_mesh, targets_on_mesh], axis=0)

# Dictionary to store features in wide format
features_dict = {'directory_id': directory_id}

print("================================================")
print("Right M1")
# Right M1
for idx, cell_idx in enumerate(right_cells):
    # Calculate M1 length
    m1p = points[pointids[cell_idx][0]]
    m1p_r = pointdata['MaximumInscribedSphereRadius'][[pointids[cell_idx][0]]][0] 
    m1d = points[pointids[cell_idx][-1]]
    m1d_r = pointdata['MaximumInscribedSphereRadius'][[pointids[cell_idx][-1]]][0] 
    m1_length = m1p_r + m1d_r + celldata['Length'][cell_idx]
    print(f"rt_m1_length{idx+1}: {m1_length * voxel_size:.3f} mm")

    # Calculate euclidean distance between the closest endpoints and the centerline points
    idx_endpoints_m1p = np.argmin(np.linalg.norm(endpoints - m1p, axis=1))
    closest_endpoints_m1p = endpoints[idx_endpoints_m1p]
    idx_endpoints_m1d = np.argmin(np.linalg.norm(endpoints - m1d, axis=1))
    closest_endpoints_m1d = endpoints[idx_endpoints_m1d]
    euclid = np.linalg.norm(closest_endpoints_m1p - closest_endpoints_m1d )
    print(f"rt_m1_length{idx+1}_euclid: {euclid * voxel_size:.3f} mm")

    # Calculate tortuosity index
    TI = m1_length / euclid
    print(f"rt_m1_length{idx+1}_ti: {TI:.3f}")

    # Calculate M1 radius
    m1_points = points[pointids[cell_idx]]
    m1_radius_arr = pointdata["MaximumInscribedSphereRadius"][pointids[cell_idx]]
    m1_radius_median = np.median(m1_radius_arr) * voxel_size
    m1_radius_q1 = np.percentile(m1_radius_arr, 25) * voxel_size
    m1_radius_q3 = np.percentile(m1_radius_arr, 75) * voxel_size
    m1_radius_iqr = (m1_radius_q3 - m1_radius_q1) * voxel_size
    m1_radius_mean = np.mean(m1_radius_arr) * voxel_size
    m1_radius_std = np.std(m1_radius_arr, ddof=1) * voxel_size
    print(f"rt_m1_radius{idx+1}_median: {m1_radius_median:.3f} mm")
    print(f"rt_m1_radius{idx+1}_q1: {m1_radius_q1:.3f} mm")
    print(f"rt_m1_radius{idx+1}_q3: {m1_radius_q3:.3f} mm")
    print(f"rt_m1_radius{idx+1}_iqr: {m1_radius_iqr:.3f} mm")
    print(f"rt_m1_radius{idx+1}_mean: {m1_radius_mean:.3f} mm")
    print(f"rt_m1_radius{idx+1}_std: {m1_radius_std:.3f} mm")
    print("================================================")
    
    # Store data in wide format
    features_dict[f'rt_m1_length{idx+1}'] = m1_length * voxel_size
    features_dict[f'rt_m1_length{idx+1}_euclid'] = euclid * voxel_size
    features_dict[f'rt_m1_length{idx+1}_ti'] = TI
    features_dict[f'rt_m1_radius{idx+1}_median'] = m1_radius_median
    features_dict[f'rt_m1_radius{idx+1}_q1'] = m1_radius_q1
    features_dict[f'rt_m1_radius{idx+1}_q3'] = m1_radius_q3
    features_dict[f'rt_m1_radius{idx+1}_iqr'] = m1_radius_iqr
    features_dict[f'rt_m1_radius{idx+1}_mean'] = m1_radius_mean
    features_dict[f'rt_m1_radius{idx+1}_std'] = m1_radius_std


print("Left M1")
# Left M1
for idx, cell_idx in enumerate(left_cells):
    # Calculate M1 length
    m1p = points[pointids[cell_idx][0]]
    m1p_r = pointdata['MaximumInscribedSphereRadius'][[pointids[cell_idx][0]]][0] 
    m1d = points[pointids[cell_idx][-1]]
    m1d_r = pointdata['MaximumInscribedSphereRadius'][[pointids[cell_idx][-1]]][0] 
    m1_length = m1p_r + m1d_r + celldata['Length'][cell_idx]
    print(f"lt_m1_length{idx+1}: {m1_length * voxel_size:.3f} mm")

    # Calculate euclidean distance between the closest endpoints and the centerline points
    idx_endpoints_m1p = np.argmin(np.linalg.norm(endpoints - m1p, axis=1))
    closest_endpoints_m1p = endpoints[idx_endpoints_m1p]
    idx_endpoints_m1d = np.argmin(np.linalg.norm(endpoints - m1d, axis=1))
    closest_endpoints_m1d = endpoints[idx_endpoints_m1d]
    euclid = np.linalg.norm(closest_endpoints_m1p - closest_endpoints_m1d )
    print(f"lt_m1_length{idx+1}_euclid: {euclid * voxel_size:.3f} mm")

    # Calculate tortuosity index
    TI = m1_length / euclid
    print(f"lt_m1_length{idx+1}_ti: {TI:.3f}")

    # Calculate M1 radius
    m1_points = points[pointids[cell_idx]]
    m1_radius_arr = pointdata["MaximumInscribedSphereRadius"][pointids[cell_idx]]
    m1_radius_median = np.median(m1_radius_arr) * voxel_size
    m1_radius_q1 = np.percentile(m1_radius_arr, 25) * voxel_size
    m1_radius_q3 = np.percentile(m1_radius_arr, 75) * voxel_size
    m1_radius_iqr = (m1_radius_q3 - m1_radius_q1) * voxel_size
    m1_radius_mean = np.mean(m1_radius_arr) * voxel_size
    m1_radius_std = np.std(m1_radius_arr, ddof=1) * voxel_size
    print(f"lt_m1_radius{idx+1}_median: {m1_radius_median:.3f} mm")
    print(f"lt_m1_radius{idx+1}_q1: {m1_radius_q1:.3f} mm")
    print(f"lt_m1_radius{idx+1}_q3: {m1_radius_q3:.3f} mm")
    print(f"lt_m1_radius{idx+1}_iqr: {m1_radius_iqr:.3f} mm")
    print(f"lt_m1_radius{idx+1}_mean: {m1_radius_mean:.3f} mm")
    print(f"lt_m1_radius{idx+1}_std: {m1_radius_std:.3f} mm")
    print("================================================")
    
    # Store data in wide format
    features_dict[f'lt_m1_length{idx+1}'] = m1_length * voxel_size
    features_dict[f'lt_m1_length{idx+1}_euclid'] = euclid * voxel_size
    features_dict[f'lt_m1_length{idx+1}_ti'] = TI
    features_dict[f'lt_m1_radius{idx+1}_median'] = m1_radius_median
    features_dict[f'lt_m1_radius{idx+1}_q1'] = m1_radius_q1
    features_dict[f'lt_m1_radius{idx+1}_q3'] = m1_radius_q3
    features_dict[f'lt_m1_radius{idx+1}_iqr'] = m1_radius_iqr
    features_dict[f'lt_m1_radius{idx+1}_mean'] = m1_radius_mean
    features_dict[f'lt_m1_radius{idx+1}_std'] = m1_radius_std

# Create DataFrame in wide format (single row)
df_features = pd.DataFrame([features_dict])

# Ensure output directory exists
output_dir = os.path.join(project_root, 'basic_morphometrics', directory_id)
utils.ensure_directory_exists(output_dir, verbose=True)

# Save to CSV
csv_path = os.path.join(output_dir, f"{directory_id}_M1_features.csv")
df_features.to_csv(csv_path, index=False)
print(f"\nFeatures saved to: {csv_path}")
