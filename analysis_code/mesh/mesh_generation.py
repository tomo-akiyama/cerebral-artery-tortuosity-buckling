import vtk
from vtk.util import numpy_support
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import os
import pymeshlab

def load_vtp(file_name):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()
    polydata = reader.GetOutput()
    return polydata

def save_vtp(vtk_data, file_name):
    # .vtpファイルに書き出す
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(file_name)
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(vtk_data)
    else:
        writer.SetInputData(vtk_data)
    writer.Write()

def np2vtk(img_np, spacing=(1,1,1), origin=(0,0,0)):
    flatten_array = img_np.ravel(order="C") # ここでtransposeを適用
    shape = np.array(img_np.shape)
    vtk_data_array = numpy_support.numpy_to_vtk(
        num_array=flatten_array,  # ndarray contains the fitting result from the points. It is a 3D array
        deep=True,
        array_type=vtk.VTK_FLOAT)
    vtk_data_array.SetName("value")
    
    # convert vessel array to poly and save as STL
    img_vtk = vtk.vtkImageData()
    img_vtk.SetDimensions(shape[::-1]) # Rmks; np.array(img3d.shape[::-1])としないとPolydataが正しい作成されない
    img_vtk.SetSpacing(spacing)
    img_vtk.SetOrigin(origin)
    img_vtk.GetPointData().SetScalars(vtk_data_array)

    return img_vtk

# VKT windowed sinc smoothing（窓関数シンクスムージング) = Taubin method
def vtk_mc_wss(img_vtk, threshold=1, iterations=30, pass_band=0.03, feature_angle=120.0, holesize=20):
    surf = vtk.vtkDiscreteMarchingCubes()
    surf.SetInputData(img_vtk)
    surf.SetValue(0, threshold)

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(surf.GetOutputPort())
    smoother.SetNumberOfIterations(iterations)  # パス数 default 20
    smoother.BoundarySmoothingOff()  # 境界スムージングオフ
    smoother.FeatureEdgeSmoothingOff()  # フィーチャーエッジスムージングオフ
    smoother.SetFeatureAngle(feature_angle)  # フィーチャーアングル(フィーチャーエッジスムージングオンの時)
    smoother.SetPassBand(pass_band)  # パスバンド 0-1
    smoother.NonManifoldSmoothingOn()  # 非多様体スムージングオン
    smoother.NormalizeCoordinatesOff()  # 座標の正規化オフ
    smoother.Update()

    fillHolesFilter = vtk.vtkFillHolesFilter()
    fillHolesFilter.SetInputData(smoother.GetOutput())
    fillHolesFilter.SetHoleSize(holesize)  # adjust the value as needed, ここでは9で初めて埋まる
    fillHolesFilter.Update()

    return fillHolesFilter.GetOutput()

# メッシュのサブディビジョン (バタフライ補間: 1回のsubdivisionで 4 倍の面となる)
def butterfly_subdivision(mesh_vtk, num_subdivision=1):
    subdivisionFilter = vtk.vtkButterflySubdivisionFilter()
    subdivisionFilter.SetInputData(mesh_vtk)
    subdivisionFilter.SetNumberOfSubdivisions(num_subdivision)  # サブディビジョンのレベルを指定
    subdivisionFilter.Update()
    return subdivisionFilter.GetOutput()

def vtk_mc(img_vtk, threshold=1): # img_vtk=vtkimagedata
    surf = vtk.vtkDiscreteMarchingCubes()
    surf.SetInputData(img_vtk)
    surf.SetValue(0, threshold)
    surf.Update()
    return surf.GetOutput()

def export_vtkpolydata_to_stl(vtk_polydata, output_filename, binary=True):
    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetInputData(vtk_polydata)
    stl_writer.SetFileName(output_filename)
    if binary:
        stl_writer.SetFileType(vtk.VTK_BINARY)
    else:
        stl_writer.SetFileType(vtk.VTK_ASCII)
    stl_writer.Write()

def export_vtkpolydata_to_ply(vtk_polydata, output_filename, binary=True):
    ply_writer = vtk.vtkPLYWriter()
    ply_writer.SetInputData(vtk_polydata)
    ply_writer.SetFileName(output_filename)
    if binary:
        ply_writer.SetFileType(vtk.VTK_BINARY)  # バイナリファイルとして保存
    else:
        ply_writer.SetFileType(vtk.VTK_ASCII)  # ASCIIファイルとして保存
    ply_writer.Write()



def load_stl_to_polydata(stl_file_path):
    # STLファイルを読み込むためのリーダーを作成
    stl_reader = vtk.vtkSTLReader()
    stl_reader.SetFileName(stl_file_path)
    
    # リーダーからポリデータを取得
    stl_reader.Update()
    poly_data = stl_reader.GetOutput()
    
    return poly_data
    
def load_ply_to_polydata(ply_file_path):
    # PLYファイルを読み込むためのリーダーを作成
    ply_reader = vtk.vtkPLYReader()
    ply_reader.SetFileName(ply_file_path)
    
    # リーダーからポリデータを取得
    ply_reader.Update()
    poly_data = ply_reader.GetOutput()
    return poly_data


    
def vol_distribution(vtkPolydata):
    # Connectity filterの作成
    connectivity = vtk.vtkConnectivityFilter()
    connectivity.SetInputData(vtkPolydata)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOn()
    connectivity.Update()

    # pyvistaで確認
    mesh = pv.wrap(connectivity.GetOutput())
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars="RegionId",
        categories=True,     # ← ここが重要
        cmap="tab20",
        show_scalar_bar=True
    )
    plotter.camera_position = 'xy'
    plotter.camera.azimuth = 180
    plotter.camera.zoom(1.2)
    plotter.show()

    # object数を確認
    n_components = connectivity.GetNumberOfExtractedRegions()
    print(f"Number of components: {n_components}")

    vol_list = []
    for i in range(n_components):
        connectivity.SetExtractionModeToSpecifiedRegions()
        connectivity.InitializeSpecifiedRegionList()
        connectivity.AddSpecifiedRegion(i)
        connectivity.Update()
    
        massProperty = vtk.vtkMassProperties()
        massProperty.SetInputData(connectivity.GetOutput())
        massProperty.Update()

        volume = massProperty.GetVolume()
        vol_list.append(volume)
        print(f"Region {i} Volume: {volume}")

    return vol_list


def get_vertices_triangles(vtk_polydata):
    # vtkPolyDataから点の座標を取得する
    points = vtk_polydata.GetPoints()
    num_points = points.GetNumberOfPoints()
    vertices = np.zeros((num_points, 3))  # X, Y, Z座標を格納するための配列を初期化
    # 各頂点の座標を配列に格納
    for i in range(num_points):
        vertices[i] = points.GetPoint(i)

    # セル（面）の情報を取得
    cells = vtk_polydata.GetPolys()  # 三角形ポリゴンを取得する場合
    cell_array = cells.GetData()
    num_cells = cells.GetNumberOfCells()
    # セルデータをNumPy配列に変換
    cell_indices = np.array(cell_array)  # VTKのセル配列をNumPy配列に変換
    # 三角形のインデックスだけを抽出
    triangles = np.reshape(cell_indices, (-1, 4))[:, 1:]  # 最初の列がポリゴンの頂点数（ここでは3）なので除外
    return vertices, triangles

def extract_object_connected_to_points(vtk_polydata, source_points, verbose=True):
    """
    source_pointsに接している、または最も近いconnected component（object）のみを抽出する
    
    Args:
        vtk_polydata: 入力のvtkPolyData（複数のconnected componentsを含む可能性がある）
        source_points: numpy array of shape (n_points, 3) - source pointsの座標
        verbose: 詳細情報を出力するかどうか
    
    Returns:
        extracted_polydata: source_pointsに最も近いconnected componentのみを含むvtkPolyData
    """
    from scipy.spatial import KDTree
    
    # 1. メッシュをconnected componentsに分割
    connectivity = vtk.vtkConnectivityFilter()
    connectivity.SetInputData(vtk_polydata)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOn()
    connectivity.Update()
    
    n_components = connectivity.GetNumberOfExtractedRegions()
    if verbose:
        print(f"Number of connected components: {n_components}")
    
    if n_components == 0:
        raise ValueError("No connected components found in the mesh")
    
    # 2. 各connected componentのRegionIdを取得
    labeled_mesh = connectivity.GetOutput()
    region_ids = labeled_mesh.GetPointData().GetArray("RegionId")
    
    # 3. メッシュの全点の座標を取得
    mesh_points = labeled_mesh.GetPoints()
    num_mesh_points = mesh_points.GetNumberOfPoints()
    mesh_coords = np.zeros((num_mesh_points, 3))
    for i in range(num_mesh_points):
        mesh_coords[i] = mesh_points.GetPoint(i)
    
    # 4. source_pointsに最も近いメッシュの点を見つける
    tree = KDTree(mesh_coords)
    distances, nearest_indices = tree.query(source_points)
    
    # 5. 最も近い点が属するRegionIdを取得
    target_region_ids = set()
    for idx in nearest_indices:
        region_id = int(region_ids.GetValue(idx))
        target_region_ids.add(region_id)
        if verbose:
            print(f"Source point nearest to mesh point {idx}, belongs to region {region_id}")
    
    if verbose:
        print(f"Target region IDs: {target_region_ids}")
    
    # 6. 対象のconnected componentのみを抽出
    connectivity.SetExtractionModeToSpecifiedRegions()
    connectivity.InitializeSpecifiedRegionList()
    for region_id in target_region_ids:
        connectivity.AddSpecifiedRegion(region_id)
    connectivity.Update()
    
    extracted_polydata = connectivity.GetOutput()
    
    # 7. 未使用の点を削除
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(extracted_polydata)
    cleaner.Update()
    
    if verbose:
        print(f"Extracted mesh: {cleaner.GetOutput().GetNumberOfPoints()} points, {cleaner.GetOutput().GetNumberOfPolys()} polygons")
    
    return cleaner.GetOutput()


def filter_by_volume(vtkPolydata, threshold_volume):
    """
    指定したボリューム閾値以下のオブジェクトを削除する関数。

    Args:
        vtkPolydata: 入力のvtkPolyData。
        threshold_volume (float): ボリュームの閾値。

    Returns:
        vtkPolyData: フィルタリングされた結果のvtkPolyData。
    """
    # Connectivity filterを適用
    connectivity = vtk.vtkConnectivityFilter()
    connectivity.SetInputData(vtkPolydata)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOn()
    connectivity.Update()

    # 出力データから各リージョンを分割
    regions_polydata = vtk.vtkPolyData()
    regions_polydata.ShallowCopy(connectivity.GetOutput())

    # リージョンごとにボリュームを計算
    region_ids_to_keep = []

    num_regions = connectivity.GetNumberOfExtractedRegions()
    for region_id in range(num_regions):
        threshold = vtk.vtkThreshold()
        threshold.SetInputData(regions_polydata)
        threshold.ThresholdBetween(region_id, region_id)
        threshold.Update()

        surface_filter = vtk.vtkDataSetSurfaceFilter()
        surface_filter.SetInputData(threshold.GetOutput())
        surface_filter.Update()

        mass_properties = vtk.vtkMassProperties()
        mass_properties.SetInputData(surface_filter.GetOutput())
        volume = mass_properties.GetVolume()

        if volume > threshold_volume:
            region_ids_to_keep.append(region_id)

    # フィルタリングされた結果をマージ
    append_filter = vtk.vtkAppendPolyData()

    for region_id in region_ids_to_keep:
        threshold = vtk.vtkThreshold()
        threshold.SetInputData(regions_polydata)
        threshold.ThresholdBetween(region_id, region_id)
        threshold.Update()

        surface_filter = vtk.vtkDataSetSurfaceFilter()
        surface_filter.SetInputData(threshold.GetOutput())
        surface_filter.Update()

        append_filter.AddInputData(surface_filter.GetOutput())

    append_filter.Update()

    # 結果のPolyDataを取得
    filtered_polydata = vtk.vtkPolyData()
    filtered_polydata.ShallowCopy(append_filter.GetOutput())

    return filtered_polydata

def generate_vessel_mesh (vessel_mc_path, 
    output_mesh_path,
    volume_threshold = 500,
    l=0.6307,
    m=-0.6732,
    s=10,
    i=2):

    # 細かいメッシュ
    # メッシュセットの作成
    ms = pymeshlab.MeshSet()
    # メッシュファイルの読み込み
    ms.load_new_mesh(vessel_mc_path)
    vertex_count1 = ms.current_mesh().vertex_number()
    print(f"Number of vertices in marching cubes mesh: {vertex_count1}")

    # Taubin smoothing (lambda=0.6307, mu=-0.6732は原著で使われている値、pass band = 0.1と同じ)
    ms.apply_coord_taubin_smoothing(lambda_=l, mu=m, stepsmoothnum=s)
    vertex_count2 = ms.current_mesh().vertex_number()
    print(f"Number of vertices in taubin smoothed mesh: {vertex_count2}")

    # Butterfly subdivision
    # iter 1で4倍に分割, iter 2では16倍に分割 → 3回繰り返すと256倍に分割
    # thresholdは特定の長さ以上のエッジを持つすべてのエッジを細分化するために使用される。0%の場合、すべてのエッジが細分化される。
    ms.meshing_surface_subdivision_butterfly(iterations = i, threshold = pymeshlab.PercentageValue(0.000000))
    vertex_count3 = ms.current_mesh().vertex_number()
    print(f"Number of vertices in post-subdivision mesh: {vertex_count3}")
    print(f"Increase rate in number of faces: {(vertex_count3/vertex_count1):2f}")

    # Save post-subdivision mesh to vessel_meshes directory as ply file
    ms.save_current_mesh(output_mesh_path)

    # vessel meshのvolumeを再計算
    vessel_mc_taubin_subdivision = load_ply_to_polydata(output_mesh_path)
    vol_list = vol_distribution(vessel_mc_taubin_subdivision)
    
    return vessel_mc_taubin_subdivision


from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
import nibabel as nib
def gifti_to_vtkpolydata(gii_file_path, nii_file_path):
    # Surface dataのGIFTIファイルを読み込み
    gii_data = nib.load(gii_file_path)

    
    # Intentコードに基づいてデータを分類する
    if gii_data.darrays[0].intent == 1009:  # NIFTI_INTENT_POINTSET
        faces = gii_data.darrays[0].data
        vertices = gii_data.darrays[1].data
    elif gii_data.darrays[0].intent == 1008:  # NIFTI_INTENT_TRIANGLE
        vertices = gii_data.darrays[0].data
        faces = gii_data.darrays[1].data
    else:
        raise ValueError("GIFTIファイルのDataArrayのIntentが不明です")


    # 元の座標系のNIfTIファイルを読み込み
    nii_data = nib.load(nii_file_path)
    # アフィン行列を計算
    affine = nii_data.affine
    # 逆アフィン行列を計算
    inverse_affine = np.linalg.inv(affine)

    # 変換行列を使用して頂点を逆変換
    # 頂点座標に1を追加して4x4行列に対応させる
    vertices_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    transformed_vertices = inverse_affine.dot(vertices_homogeneous.T).T[:, :3]

    # vtkの座標系に変換 xyz -> zyx
    transformed_vertices = transformed_vertices[:, [2, 1, 0]]

    # float32にキャスト
    transformed_vertices = transformed_vertices.astype(np.float32)


    # VTK Pointsオブジェクトに変換
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(transformed_vertices ))

    # VTK CellArrayオブジェクトに変換
    faces_vtk = np.hstack((np.full((faces.shape[0], 1), 3), faces)).astype(np.int64)
    cells = vtk.vtkCellArray()
    cells.SetCells(faces.shape[0], numpy_to_vtkIdTypeArray(faces_vtk))

    # VTK PolyDataオブジェクトの作成
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)

    return polydata