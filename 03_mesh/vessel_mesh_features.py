#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: vessel mesh features."""

import numpy as np
import pyvista as pv
import vtk


def compute_edge_lengths(polydata):
    mesh = pv.wrap(polydata)
    if mesh.n_cells == 0:
        return np.array([], dtype=float)

             
    edges = mesh.extract_all_edges()
    if edges.n_cells == 0:
        return np.array([], dtype=float)

                                                      
    lines = edges.lines.reshape(-1, 3)[:, 1:]  # shape = (n_edges, 2)

                             
    lines_sorted = np.sort(lines, axis=1)
    _, unique_idx = np.unique(lines_sorted, axis=0, return_index=True)
    unique_lines = lines_sorted[np.sort(unique_idx)]

    pts = edges.points
    p0 = pts[unique_lines[:, 0]]
    p1 = pts[unique_lines[:, 1]]

    return np.linalg.norm(p1 - p0, axis=1)


def summarize_edge_lengths(edge_lengths):
    edge_lengths = np.asarray(edge_lengths, dtype=float)
    n = edge_lengths.size
    if n == 0:
        return {
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "iqr": np.nan,
        }
    q1 = float(np.percentile(edge_lengths, 25))
    q3 = float(np.percentile(edge_lengths, 75))
    return {
        "mean": float(edge_lengths.mean()),
        "sd": float(edge_lengths.std(ddof=1)) if n >= 2 else np.nan,
        "median": float(np.median(edge_lengths)),
        "iqr": q3 - q1,
    }


def compute_mesh_volume(polydata):
                  
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(polydata)
    tri.Update()

          
    mass = vtk.vtkMassProperties()
    mass.SetInputData(tri.GetOutput())
    mass.Update()

    volume = mass.GetVolume()
    area = mass.GetSurfaceArea()

    return volume, area


def kmeans_1d_two_clusters(x, n_iter=50):
    x = np.asarray(x).reshape(-1, 1)
                   
    c1, c2 = np.percentile(x, [33, 66]).reshape(2, 1)
    for _ in range(n_iter):
            
        d1 = np.abs(x - c1)
        d2 = np.abs(x - c2)
        labels = (d2 < d1).astype(int)  # 0 or 1
            
        if np.any(labels == 0):
            c1_new = x[labels == 0].mean()
        else:
            c1_new = c1
        if np.any(labels == 1):
            c2_new = x[labels == 1].mean()
        else:
            c2_new = c2
                
        if np.isclose(c1, c1_new).all() and np.isclose(c2, c2_new).all():
            break
        c1, c2 = np.array([[c1_new]]), np.array([[c2_new]])
    centers = np.array([c1.item(), c2.item()])
    labels = labels.ravel()
    return labels, centers


def classify_centerline_left_right_by_kmeans(points, pointids, n_iter=100, verbose=True):
    xs = points[:, 0].astype(float)
    
                      
    cell_median_x = []
    for cell in pointids:
        xm = np.median(xs[np.asarray(cell, dtype=int)])
        cell_median_x.append(xm)
    cell_median_x = np.asarray(cell_median_x)
    
                   
    labels, centers = kmeans_1d_two_clusters(cell_median_x, n_iter=n_iter)
    
                            
    right_label = int(np.argmax(centers))
    left_label = 1 - right_label
    
                     
    is_right_cell = (labels == right_label)
    is_left_cell = (labels == left_label)

                                
    threshold_x = centers.mean()

                                
    is_right_point = xs >= threshold_x
    is_left_point  = xs <  threshold_x
        
              
    right_cells = np.where(is_right_cell)[0]
    left_cells = np.where(is_left_cell)[0]
    
    if verbose:
        print(f"右クラスタ中心 ≈ {centers[right_label]:.2f} / 左クラスタ中心 ≈ {centers[left_label]:.2f}")
        print(f"K-meansしきい値（左右判定用）≈ {threshold_x:.2f}")
        print(f"右セル数: {is_right_cell.sum()}, 左セル数: {is_left_cell.sum()}")
        print(f"右ポイント数: {is_right_point.sum()}, 左ポイント数: {is_left_point.sum()}")
        print(f"右セルID（{right_cells.size}本）: {right_cells.tolist()}")
        print(f"左セルID（{left_cells.size}本）: {left_cells.tolist()}")
    
    return right_cells, left_cells, threshold_x, is_right_point, is_left_point


def split_mesh_left_right(polydata, threshold_x, verbose=True):
          
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(polydata)
    tri.Update()
    poly = tri.GetOutput()
    
               
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(poly)
    conn.SetExtractionModeToAllRegions()
    conn.ColorRegionsOn()
    conn.Update()
    
    n_regions = conn.GetNumberOfExtractedRegions()
    if verbose:
        print("n_regions =", n_regions)
    
    if n_regions < 2:
        raise ValueError(f"連結成分数が2個未満（{n_regions}個）。左右分離に失敗した。")
    
    labeled = conn.GetOutput()
    
    def extract_region(region_id):
        f = vtk.vtkConnectivityFilter()
        f.SetInputData(labeled)
        f.SetExtractionModeToSpecifiedRegions()
        f.InitializeSpecifiedRegionList()
        f.AddSpecifiedRegion(region_id)
        f.Update()
        return f.GetOutput()
    
                         
    def get_region_size(region_id):
        region = extract_region(region_id)
        return region.GetNumberOfPolys()
    
                                                           
    def clean_poly(poly):
        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(poly)
        clean.Update()
        return clean.GetOutput()
    
                       
    def get_mean_x(polydata):
        n_points = polydata.GetNumberOfPoints()
        if n_points == 0:
            return 0.0
        x_coords = []
        for i in range(n_points):
            point = polydata.GetPoint(i)
            x_coords.append(point[0])
        return np.mean(x_coords)
    
                    
    right_regions = []
    left_regions = []
    right_center_x_list = []
    left_center_x_list = []
    
    for region_id in range(n_regions):
        region = extract_region(region_id)
        region = clean_poly(region)
        mean_x = get_mean_x(region)
        
                                        
        if mean_x > threshold_x:
            right_regions.append(region)
            right_center_x_list.append(mean_x)
        else:
            left_regions.append(region)
            left_center_x_list.append(mean_x)
        
        if verbose:
            print(f"Region {region_id}: mean_x = {mean_x:.2f}, {'right' if mean_x > threshold_x else 'left'}")
    
    if verbose:
        print(f"右側の連結成分数: {len(right_regions)}, 左側の連結成分数: {len(left_regions)}")
    
    if len(right_regions) == 0 or len(left_regions) == 0:
        raise ValueError(f"左右の分類に失敗しました。右側: {len(right_regions)}個, 左側: {len(left_regions)}個")
    
                 
    if len(right_regions) == 1:
        right_vessel = right_regions[0]
    else:
        append_right = vtk.vtkAppendPolyData()
        for region in right_regions:
            append_right.AddInputData(region)
        append_right.Update()
        right_vessel = clean_poly(append_right.GetOutput())
    
                 
    if len(left_regions) == 1:
        left_vessel = left_regions[0]
    else:
        append_left = vtk.vtkAppendPolyData()
        for region in left_regions:
            append_left.AddInputData(region)
        append_left.Update()
        left_vessel = clean_poly(append_left.GetOutput())
    
                    
    right_center_x = np.mean(right_center_x_list) if right_center_x_list else 0.0
    left_center_x = np.mean(left_center_x_list) if left_center_x_list else 0.0
    
    if verbose:
        print(f"右側の平均中心x座標: {right_center_x:.2f}, 左側の平均中心x座標: {left_center_x:.2f}")
    
    return right_vessel, left_vessel, right_center_x, left_center_x

