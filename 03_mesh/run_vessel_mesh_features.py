#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract and save vessel mesh features
"""

########################################################
# ===== USER SETTINGS =====
directory_id = "IXI600-HH-2660"

voxel_size = 0.5  # mm
########################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add the path to make the utils module importable
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
utils_path = os.path.join(project_root, 'lib')
sys.path.insert(0, os.path.join(project_root, 'lib'))  # Add analysis_code to path for package imports
sys.path.append(utils_path)

import pyvista as pv
from vessel_mesh_features import (
    compute_edge_lengths,
    compute_mesh_volume,
    classify_centerline_left_right_by_kmeans,
    split_mesh_left_right,
    summarize_edge_lengths,
)
import utils
import mesh_generation
import centerline_extraction


def calculate_and_save_vessel_mesh_features(
    vessel_mesh_vtk,
    vessel_centerlines_vtk,
    voxel_size,
    directory_id,
    csv_path=None,
    output_dir=None
):
    """
    Extract vessel mesh features and save them to a csv file.

    Args:
    vessel_mesh_vtk : vtk.vtkPolyData, vessel mesh
    vessel_centerlines_vtk : vtkPolyData, vessel centerlines
    voxel_size : float, mm
    directory_id : str
    csv_path : str, optional, path to the csv file
    output_dir : str, optional, path to the output directory
    """

    case_id = directory_id[3:6].lstrip("0")
    case_id = str(case_id)

    n_faces = vessel_mesh_vtk.GetNumberOfPolys()
    n_vertices = vessel_mesh_vtk.GetNumberOfPoints()
    edge_lengths_vox = compute_edge_lengths(vessel_mesh_vtk)
    volume_vox, area_vox = compute_mesh_volume(vessel_mesh_vtk)

    celldata, pointids, points, pointdata = centerline_extraction.vmtk_convert2numpy(vessel_centerlines_vtk)

    print(f"total length: {celldata['Length'].sum() * voxel_size:.3f} mm")
    print(f"number of segments: {len(pointids)}")
    print(f"number of faces: {n_faces}")
    print(f"number of vertices: {n_vertices}")
    print(f"number of edges: {edge_lengths_vox.size}")
    print(f"edge length median: {np.median(edge_lengths_vox)* voxel_size:.3f} mm")
    print(f"edge length mean: {edge_lengths_vox.mean() * voxel_size:.3f} mm")
    print(f"volume: {volume_vox * (voxel_size ** 3):.2f} mm^3")
    print(f"surface area: {area_vox * (voxel_size ** 2):.2f} mm^2")

                 
    right_cells, left_cells, threshold_x, is_right_point, is_left_point = classify_centerline_left_right_by_kmeans(
        points, pointids, n_iter=100, verbose=True
    )

                
    right_vessel, left_vessel, right_center_x, left_center_x = split_mesh_left_right(
        vessel_mesh_vtk, threshold_x, verbose=True
    )

    rt_vessel_length =  celldata['Length'][right_cells].sum()
    lt_vessel_length =  celldata['Length'][left_cells].sum()

    n_faces_right = right_vessel.GetNumberOfPolys()
    n_faces_left = left_vessel.GetNumberOfPolys()
    n_vertices_right = right_vessel.GetNumberOfPoints()
    n_vertices_left = left_vessel.GetNumberOfPoints()
    edge_lengths_vox_right = compute_edge_lengths(right_vessel)
    edge_lengths_vox_left = compute_edge_lengths(left_vessel)
    volume_vox_right, area_vox_right = compute_mesh_volume(right_vessel)
    volume_vox_left, area_vox_left = compute_mesh_volume(left_vessel)

    print("================================================")
    print(f"Rt vessel total length: {rt_vessel_length * voxel_size:.3f} mm")
    print(f"Rt number of segments: {right_cells.size}")
    print(f"Rt number of faces: {n_faces_right}")
    print(f"Rt number of vertices: {n_vertices_right}")
    print(f"Rt number of edges: {edge_lengths_vox_right.size}")
    print(f"Rt edge length median: {np.median(edge_lengths_vox_right)* voxel_size:.3f} mm")
    print(f"Rt edge length mean: {edge_lengths_vox_right.mean() * voxel_size:.3f} mm")
    print(f"Rt volume: {volume_vox_right * (voxel_size ** 3):.2f} mm^3")
    print(f"Rt surface area: {area_vox_right * (voxel_size ** 2):.2f} mm^2")
    print("================================================")
    print(f"Lt vessel total length: {lt_vessel_length * voxel_size:.3f} mm")
    print(f"Lt number of segments: {left_cells.size}")
    print(f"Lt number of faces: {n_faces_left}")
    print(f"Lt number of vertices: {n_vertices_left}")
    print(f"Lt number of edges: {edge_lengths_vox_left.size}")
    print(f"Lt edge length median: {np.median(edge_lengths_vox_left)* voxel_size:.3f} mm")
    print(f"Lt edge length mean: {edge_lengths_vox_left.mean() * voxel_size:.3f} mm")
    print(f"Lt volume: {volume_vox_left * (voxel_size ** 3):.2f} mm^3")
    print(f"Lt surface area: {area_vox_left * (voxel_size ** 2):.2f} mm^2")


                                          
    total_vessel_length_vox = celldata["Length"].sum()
    edge_stats_vox = summarize_edge_lengths(edge_lengths_vox)

    mesh_data = {
        "IXI_ID": case_id,

                                                  
        "total_n_segments": len(pointids),
        "total_n_edges": edge_lengths_vox.size,
        "total_n_faces": n_faces,
        "total_n_vertices": n_vertices,
        "total_mesh_volume_vox": volume_vox,
        "total_mesh_volume_mm3": volume_vox * (voxel_size ** 3),
        "total_surface_area_vox": area_vox,
        "total_surface_area_mm2": area_vox * (voxel_size ** 2),
        "total_vessel_length_vox": total_vessel_length_vox,
        "total_vessel_length_mm": total_vessel_length_vox * voxel_size,
        "total_edge_length_mean_vox": edge_stats_vox["mean"],
        "total_edge_length_sd_vox": edge_stats_vox["sd"],
        "total_edge_length_median_vox": edge_stats_vox["median"],
        "total_edge_length_iqr_vox": edge_stats_vox["iqr"],
        "total_edge_length_mean_mm": edge_stats_vox["mean"] * voxel_size,
        "total_edge_length_sd_mm": edge_stats_vox["sd"] * voxel_size,
        "total_edge_length_median_mm": edge_stats_vox["median"] * voxel_size,
        "total_edge_length_iqr_mm": edge_stats_vox["iqr"] * voxel_size,

        # right
        "rt_n_segments": right_cells.size,
        "rt_n_edges": edge_lengths_vox_right.size,
        "rt_n_faces": n_faces_right,
        "rt_n_vertices": n_vertices_right,
        "rt_mesh_volume_vox": volume_vox_right,
        "rt_mesh_volume_mm3": volume_vox_right * (voxel_size ** 3),
        "rt_surface_area_vox": area_vox_right,
        "rt_surface_area_mm2": area_vox_right * (voxel_size ** 2),
        "rt_vessel_length_vox": rt_vessel_length,
        "rt_vessel_length_mm": rt_vessel_length * voxel_size,

        # left
        "lt_n_segments": left_cells.size,
        "lt_n_edges": edge_lengths_vox_left.size,
        "lt_n_faces": n_faces_left,
        "lt_n_vertices": n_vertices_left,
        "lt_mesh_volume_vox": volume_vox_left,
        "lt_mesh_volume_mm3": volume_vox_left * (voxel_size ** 3),
        "lt_surface_area_vox": area_vox_left,
        "lt_surface_area_mm2": area_vox_left * (voxel_size ** 2),
        "lt_vessel_length_vox": lt_vessel_length,
        "lt_vessel_length_mm": lt_vessel_length * voxel_size,
    }

                    
    if csv_path is not None and directory_id is not None:
                      
        if os.path.exists(csv_path):
            df_csv = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"csvファイルが見つからない: {csv_path}")
        
                               
        mask = df_csv['directory_id'] == directory_id
        matching_rows = df_csv[mask]
        
                                  
        for key, value in mesh_data.items():
            if key != "IXI_ID":
                df_csv.loc[mask, key] = value
        
                    
        df_csv.to_csv(csv_path, index=False)
        print(f"csvファイルを更新しました: {csv_path}")
        
        return df_csv[mask]
    
    
                
    if output_dir is None:
        raise ValueError("output_dir must be specified when csv_path is None")
    
                         
    utils.ensure_directory_exists(output_dir, verbose=True)
    
    df_vessel_mesh_features = pd.DataFrame([mesh_data])
    csv_file_path = f"{output_dir}/{directory_id}_vessel_mesh_features.csv"
    df_vessel_mesh_features.to_csv(csv_file_path, index=None)
    print(f"csv file saved: {csv_file_path}")
    return df_vessel_mesh_features


if __name__ == "__main__":
    vessel_mesh_dir = os.path.join(project_root, 'data', 'preprocessed', 'MRA_preprocessed', 'vessel_meshes', directory_id)
    vessel_mesh_path = os.path.join(vessel_mesh_dir, f"{directory_id}_vessel_mesh.ply")
    vessel_mesh_vtk = mesh_generation.load_ply_to_polydata(vessel_mesh_path)

    vessel_centerlines_dir = os.path.join(project_root, 'data', 'derived', 'vessel_centerlines', directory_id)
    vessel_centerlines_path = os.path.join(vessel_centerlines_dir, f"{directory_id}_merged_centerlines.vtp")
    vessel_centerlines_vtk = centerline_extraction.load_vtp(vessel_centerlines_path)

                                             
    output_dir = vessel_mesh_dir
    utils.ensure_directory_exists(output_dir, verbose=True)

    df_vessel_mesh_features = calculate_and_save_vessel_mesh_features(
        vessel_mesh_vtk=vessel_mesh_vtk,
        vessel_centerlines_vtk=vessel_centerlines_vtk,
        voxel_size=voxel_size,
        directory_id=directory_id,
        csv_path=None,
        output_dir=output_dir
    )
    
    print(f"Vessel mesh features extraction and saving completed: {df_vessel_mesh_features}")

