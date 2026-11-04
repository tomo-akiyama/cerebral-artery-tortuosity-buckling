"""Vessel mesh generation utilities (VTK / PyMeshLab)."""

"""Vessel mesh generation utilities (VTK / PyMeshLab)."""
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
                   
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(file_name)
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(vtk_data)
    else:
        writer.SetInputData(vtk_data)
    writer.Write()

def np2vtk(img_np, spacing=(1,1,1), origin=(0,0,0)):
    flatten_array = img_np.ravel(order="C")                  
    shape = np.array(img_np.shape)
    vtk_data_array = numpy_support.numpy_to_vtk(
        num_array=flatten_array,  # ndarray contains the fitting result from the points. It is a 3D array
        deep=True,
        array_type=vtk.VTK_FLOAT)
    vtk_data_array.SetName("value")
    
    # convert vessel array to poly and save as STL
    img_vtk = vtk.vtkImageData()
    img_vtk.SetDimensions(shape[::-1])                                                           
    img_vtk.SetSpacing(spacing)
    img_vtk.SetOrigin(origin)
    img_vtk.GetPointData().SetScalars(vtk_data_array)

    return img_vtk

# Marching cubes
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
        ply_writer.SetFileType(vtk.VTK_BINARY)                 
    else:
        ply_writer.SetFileType(vtk.VTK_ASCII)                  
    ply_writer.Write()



def load_stl_to_polydata(stl_file_path):
                            
    stl_reader = vtk.vtkSTLReader()
    stl_reader.SetFileName(stl_file_path)
    
                    
    stl_reader.Update()
    poly_data = stl_reader.GetOutput()
    
    return poly_data
    
def load_ply_to_polydata(ply_file_path):
                            
    ply_reader = vtk.vtkPLYReader()
    ply_reader.SetFileName(ply_file_path)
    
                    
    ply_reader.Update()
    poly_data = ply_reader.GetOutput()
    return poly_data


    
def vol_distribution(vtkPolydata):
                          
    connectivity = vtk.vtkConnectivityFilter()
    connectivity.SetInputData(vtkPolydata)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOn()
    connectivity.Update()

                
    mesh = pv.wrap(connectivity.GetOutput())
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars="RegionId",
        categories=True,              
        cmap="tab20",
        show_scalar_bar=True
    )
    plotter.camera_position = 'xy'
    plotter.camera.azimuth = 180
    plotter.camera.zoom(1.2)
    plotter.show()

                
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
                            
    points = vtk_polydata.GetPoints()
    num_points = points.GetNumberOfPoints()
    vertices = np.zeros((num_points, 3))                           
                  
    for i in range(num_points):
        vertices[i] = points.GetPoint(i)

                 
    cells = vtk_polydata.GetPolys()                  
    cell_array = cells.GetData()
    num_cells = cells.GetNumberOfCells()
                      
    cell_indices = np.array(cell_array)                       
                     
    triangles = np.reshape(cell_indices, (-1, 4))[:, 1:]                             
    return vertices, triangles

def extract_object_connected_to_points(vtk_polydata, source_points, verbose=True):
    from scipy.spatial import KDTree
    
                                     
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
    
                                         
    labeled_mesh = connectivity.GetOutput()
    region_ids = labeled_mesh.GetPointData().GetArray("RegionId")
    
                      
    mesh_points = labeled_mesh.GetPoints()
    num_mesh_points = mesh_points.GetNumberOfPoints()
    mesh_coords = np.zeros((num_mesh_points, 3))
    for i in range(num_mesh_points):
        mesh_coords[i] = mesh_points.GetPoint(i)
    
                                      
    tree = KDTree(mesh_coords)
    distances, nearest_indices = tree.query(source_points)
    
                             
    target_region_ids = set()
    for idx in nearest_indices:
        region_id = int(region_ids.GetValue(idx))
        target_region_ids.add(region_id)
        if verbose:
            print(f"Source point nearest to mesh point {idx}, belongs to region {region_id}")
    
    if verbose:
        print(f"Target region IDs: {target_region_ids}")
    
                                    
    connectivity.SetExtractionModeToSpecifiedRegions()
    connectivity.InitializeSpecifiedRegionList()
    for region_id in target_region_ids:
        connectivity.AddSpecifiedRegion(region_id)
    connectivity.Update()
    
    extracted_polydata = connectivity.GetOutput()
    
                 
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(extracted_polydata)
    cleaner.Update()
    
    if verbose:
        print(f"Extracted mesh: {cleaner.GetOutput().GetNumberOfPoints()} points, {cleaner.GetOutput().GetNumberOfPolys()} polygons")
    
    return cleaner.GetOutput()


def filter_by_volume(vtkPolydata, threshold_volume):
                            
    connectivity = vtk.vtkConnectivityFilter()
    connectivity.SetInputData(vtkPolydata)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOn()
    connectivity.Update()

                      
    regions_polydata = vtk.vtkPolyData()
    regions_polydata.ShallowCopy(connectivity.GetOutput())

                      
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

             
                
    ms = pymeshlab.MeshSet()
                   
    ms.load_new_mesh(vessel_mc_path)
    vertex_count1 = ms.current_mesh().vertex_number()
    print(f"Number of vertices in marching cubes mesh: {vertex_count1}")

                                                                                
    ms.apply_coord_taubin_smoothing(lambda_=l, mu=m, stepsmoothnum=s)
    vertex_count2 = ms.current_mesh().vertex_number()
    print(f"Number of vertices in taubin smoothed mesh: {vertex_count2}")

    # Butterfly subdivision
                                                   
                                                                         
    ms.meshing_surface_subdivision_butterfly(iterations = i, threshold = pymeshlab.PercentageValue(0.000000))
    vertex_count3 = ms.current_mesh().vertex_number()
    print(f"Number of vertices in post-subdivision mesh: {vertex_count3}")
    print(f"Increase rate in number of faces: {(vertex_count3/vertex_count1):2f}")

    # Save post-subdivision mesh to vessel_meshes directory as ply file
    ms.save_current_mesh(output_mesh_path)
    
                            
    vessel_mc_taubin_subdivision = load_ply_to_polydata(output_mesh_path)
    vol_list = vol_distribution(vessel_mc_taubin_subdivision)
    
    return vessel_mc_taubin_subdivision


from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
import nibabel as nib
def gifti_to_vtkpolydata(gii_file_path, nii_file_path):
                                 
    gii_data = nib.load(gii_file_path)

    
                            
    if gii_data.darrays[0].intent == 1009:  # NIFTI_INTENT_POINTSET
        faces = gii_data.darrays[0].data
        vertices = gii_data.darrays[1].data
    elif gii_data.darrays[0].intent == 1008:  # NIFTI_INTENT_TRIANGLE
        vertices = gii_data.darrays[0].data
        faces = gii_data.darrays[1].data
    else:
        raise ValueError("Unknown Intent on GIFTI DataArray")


                          
    nii_data = nib.load(nii_file_path)
               
    affine = nii_data.affine
                
    inverse_affine = np.linalg.inv(affine)

                     
                            
    vertices_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    transformed_vertices = inverse_affine.dot(vertices_homogeneous.T).T[:, :3]

                           
    transformed_vertices = transformed_vertices[:, [2, 1, 0]]

                  
    transformed_vertices = transformed_vertices.astype(np.float32)


                         
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(transformed_vertices ))

                            
    faces_vtk = np.hstack((np.full((faces.shape[0], 1), 3), faces)).astype(np.int64)
    cells = vtk.vtkCellArray()
    cells.SetCells(faces.shape[0], numpy_to_vtkIdTypeArray(faces_vtk))

                           
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)

    return polydata