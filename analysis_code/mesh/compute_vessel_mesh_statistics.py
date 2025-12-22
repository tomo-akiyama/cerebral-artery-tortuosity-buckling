#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vessel_mesh_statistics.py 専用のユーティリティ関数
"""

import numpy as np
import pyvista as pv
import vtk


def compute_edge_lengths(polydata):
    """
    vtkPolyData から全エッジ長を抽出して np.ndarray を返す関数
    長さの単位は polydata の座標系（＝今なら voxel 単位）のまま
    """
    mesh = pv.wrap(polydata)

    # 全エッジを抽出
    edges = mesh.extract_all_edges()

    # エッジの頂点インデックス（PyVista形式: [2, i, j, 2, k, l, ...]）
    lines = edges.lines.reshape(-1, 3)[:, 1:]  # shape = (n_edges, 2)

    pts = edges.points
    p0 = pts[lines[:, 0]]
    p1 = pts[lines[:, 1]]

    # 各エッジのユークリッド距離
    edge_lengths = np.linalg.norm(p1 - p0, axis=1)

    return edge_lengths


def compute_mesh_volume(polydata):
    """
    メッシュの体積と表面積を計算
    """
    # まず全ポリゴンを三角形化
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(polydata)
    tri.Update()

    # 体積計算
    mass = vtk.vtkMassProperties()
    mass.SetInputData(tri.GetOutput())
    mass.Update()

    volume = mass.GetVolume()
    area = mass.GetSurfaceArea()

    return volume, area


def kmeans_1d_two_clusters(x, n_iter=50):
    """
    1次元データを2つのクラスタに分類するK-meansアルゴリズム
    
    Parameters
    ----------
    x : array-like
        1次元データ配列
    n_iter : int
        最大反復回数
    
    Returns
    -------
    labels : ndarray
        各データポイントのクラスタラベル (0 or 1)
    centers : ndarray
        2つのクラスタの中心座標
    """
    x = np.asarray(x).reshape(-1, 1)
    # 初期中心：分位点でざっくり
    c1, c2 = np.percentile(x, [33, 66]).reshape(2, 1)
    for _ in range(n_iter):
        # 割当
        d1 = np.abs(x - c1)
        d2 = np.abs(x - c2)
        labels = (d2 < d1).astype(int)  # 0 or 1
        # 更新
        if np.any(labels == 0):
            c1_new = x[labels == 0].mean()
        else:
            c1_new = c1
        if np.any(labels == 1):
            c2_new = x[labels == 1].mean()
        else:
            c2_new = c2
        # 収束チェック
        if np.isclose(c1, c1_new).all() and np.isclose(c2, c2_new).all():
            break
        c1, c2 = np.array([[c1_new]]), np.array([[c2_new]])
    centers = np.array([c1.item(), c2.item()])
    labels = labels.ravel()
    return labels, centers


def classify_left_right_by_kmeans(points, pointids, n_iter=100, verbose=True):
    """
    中心線セルをK-meansで左右に分類する
    
    Parameters
    ----------
    points : array-like, shape (N, 3)
        ポイントの座標
    pointids : list-like, shape (n_cells,)
        各セルの頂点インデックス列
    n_iter : int
        K-meansの最大反復回数
    verbose : bool
        詳細情報を出力するかどうか
    
    Returns
    -------
    is_right_cell : ndarray, shape (n_cells,)
        各セルが右側かどうかのブール配列（pointids[i]で左右を判別可能）
    right_cells : ndarray
        右側セルのインデックス配列
    left_cells : ndarray
        左側セルのインデックス配列
    threshold_x : float
        K-meansの2つの中心の中点（左右判定用しきい値）
    centers : ndarray
        2つのクラスタの中心座標
    """
    xs = points[:, 0].astype(float)
    
    # 各セルの代表 x（中央値）を計算
    cell_median_x = []
    for cell in pointids:
        xm = np.median(xs[np.asarray(cell, dtype=int)])
        cell_median_x.append(xm)
    cell_median_x = np.asarray(cell_median_x)
    
    # K-meansで左右に分類
    labels, centers = kmeans_1d_two_clusters(cell_median_x, n_iter=n_iter)
    
    # 大きい中心を「右」、小さい中心を「左」と定義
    right_label = int(np.argmax(centers))
    left_label = 1 - right_label
    
    # 結果：cellごとの左右フラグ（pointids[i]で左右を判別するために保持）
    is_right_cell = (labels == right_label)
    is_left_cell = (labels == left_label)
    
    # K-meansの2つの中心の中点をしきい値として使用
    threshold_x = centers.mean()
    
    # セルIDのリスト
    right_cells = np.where(is_right_cell)[0]
    left_cells = np.where(is_left_cell)[0]
    
    if verbose:
        print(f"右クラスタ中心 ≈ {centers[right_label]:.2f} / 左クラスタ中心 ≈ {centers[left_label]:.2f}")
        print(f"K-meansしきい値（左右判定用）≈ {threshold_x:.2f}")
        print(f"右セル数: {is_right_cell.sum()}, 左セル数: {is_left_cell.sum()}")
        print(f"右セルID（{right_cells.size}本）: {right_cells.tolist()}")
        print(f"左セルID（{left_cells.size}本）: {left_cells.tolist()}")
    
    return is_right_cell, right_cells, left_cells, threshold_x, centers


def split_mesh_left_right(polydata, threshold_x, verbose=True):
    """
    メッシュをK-meansのしきい値を使って左右に分離する
    
    Parameters
    ----------
    polydata : vtk.vtkPolyData
        血管メッシュ
    threshold_x : float
        K-meansのしきい値（左右判定用）
    verbose : bool
        詳細情報を出力するかどうか
    
    Returns
    -------
    right_vessel : vtk.vtkPolyData
        右側のメッシュ
    left_vessel : vtk.vtkPolyData
        左側のメッシュ
    """
    # 三角形化
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(polydata)
    tri.Update()
    poly = tri.GetOutput()
    
    # 連結成分を全て抽出
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(poly)
    conn.SetExtractionModeToAllRegions()
    conn.ColorRegionsOn()
    conn.Update()
    
    n_regions = conn.GetNumberOfExtractedRegions()
    if verbose:
        print("n_regions =", n_regions)
    
    if n_regions < 2:
        raise ValueError(f"連結成分数が2個未満です（{n_regions}個）。左右分離に失敗しました。")
    
    labeled = conn.GetOutput()
    
    def extract_region(region_id):
        f = vtk.vtkConnectivityFilter()
        f.SetInputData(labeled)
        f.SetExtractionModeToSpecifiedRegions()
        f.InitializeSpecifiedRegionList()
        f.AddSpecifiedRegion(region_id)
        f.Update()
        return f.GetOutput()
    
    # 各連結成分のサイズ（ポリゴン数）を計算
    def get_region_size(region_id):
        region = extract_region(region_id)
        return region.GetNumberOfPolys()
    
    # 全ての連結成分のサイズを取得
    region_sizes = [(i, get_region_size(i)) for i in range(n_regions)]
    # サイズでソート（大きい順）
    region_sizes.sort(key=lambda x: x[1], reverse=True)
    
    if verbose and n_regions > 2:
        print(f"警告: 連結成分が{n_regions}個検出されました。最も大きな2つを選択します。")
        print(f"各連結成分のサイズ: {[(i, size) for i, size in region_sizes]}")
    
    # 最も大きな2つの連結成分を選択
    largest_region_id = region_sizes[0][0]
    second_largest_region_id = region_sizes[1][0]
    
    part0 = extract_region(largest_region_id)
    part1 = extract_region(second_largest_region_id)
    
    # vtkConnectivityFilterは使われていない Points を保持したまま出力することがある
    def clean_poly(poly):
        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(poly)
        clean.Update()
        return clean.GetOutput()
    
    part0 = clean_poly(part0)
    part1 = clean_poly(part1)
    
    # メッシュの各点のx座標の平均を取得
    def get_mean_x(polydata):
        """メッシュの全点のx座標の平均を取得"""
        n_points = polydata.GetNumberOfPoints()
        if n_points == 0:
            return 0.0
        x_coords = []
        for i in range(n_points):
            point = polydata.GetPoint(i)
            x_coords.append(point[0])
        return np.mean(x_coords)
    
    x0_part = get_mean_x(part0)
    x1_part = get_mean_x(part1)
    if verbose:
        print("mean x: part0 =", x0_part, "part1 =", x1_part)
    
    # K-meansのしきい値を使って左右を割り当て
    # しきい値より大きいx座標を持つ方を右、小さい方を左とする
    if x0_part > threshold_x and x1_part < threshold_x:
        right_vessel, left_vessel = part0, part1
        right_center_x, left_center_x = x0_part, x1_part
    elif x1_part > threshold_x and x0_part < threshold_x:
        right_vessel, left_vessel = part1, part0
        right_center_x, left_center_x = x1_part, x0_part
    else:
        # どちらもしきい値の同じ側にある場合（稀なケース）、x座標の大小で判定
        if x0_part > x1_part:
            right_vessel, left_vessel = part0, part1
            right_center_x, left_center_x = x0_part, x1_part
        else:
            right_vessel, left_vessel = part1, part0
            right_center_x, left_center_x = x1_part, x0_part
        if verbose:
            print(f"警告: 両方の連結成分がしきい値の同じ側にあったため、x座標の大小で左右を判定した。")
    
    return right_vessel, left_vessel, right_center_x, left_center_x

