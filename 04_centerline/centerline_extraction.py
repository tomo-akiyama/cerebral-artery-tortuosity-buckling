"""VMTK wrappers for centerline extraction and processing."""
import numpy as np
from vmtk import vmtkscripts
from vmtk import vmtkcontribscripts
import vtk
from vtk.util import numpy_support
import os

import math

from matplotlib import pyplot as plt


# The Vascular Modeling Toolkit (VMTK): http://www.vmtk.org/index.html

def load_vtp(file_path):
    # Create a reader for vtp file
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    # Output a vtkPolyData object
    vtkpolydata = reader.GetOutput()
    return vtkpolydata

def save_vtp(vtkpolydata, file_path):
    # Save a vtkPolyData object as vtp file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(file_path)
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(vtkpolydata)
    else:
        writer.SetInputData(vtkpolydata)
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


"""
def vmtk_extract_centerline(mesh, seedselector="pickpoint"):
    #==============================================================================
    # args: mesh: 3D mesh (vtkPolydata)
    #       seedselector: "pickpoint","openprofiles","carotidprofiles","profileidlist","idlist","pointlist"
    # returns: centerlines (vtkPolyData)
    #==============================================================================
    # Extract centerline
    centerline_extractor = vmtkscripts.vmtkCenterlines()
    centerline_extractor.Surface = mesh
    # Define source and target points
    centerline_extractor.SeedSelectorName = seedselector
    centerline_extractor.Execute()
    centerlines = centerline_extractor.Centerlines
    return centerlines
"""

def vmtk_extract_centerline(file_name, sources, targets, step_length=0.1, seedselector="pointlist"):
    # vmtk_extract_centerline(mesh, sources, targets, seedselector="pointlist"):
    #==============================================================================
    # args: mesh: 3D mesh (vtkPolydata)
    #       sources: 3D coordinates of source points (numpy.ndarray)
    #       targets: 3D coordinates of target points (numpy.ndarray)
    #       seedselector: Options include "pickpoint", "openprofiles", "carotidprofiles", "profileidlist", "idlist" and "pointlist".
    # returns: centerlines (vtkPolyData)
    #==============================================================================
    source_points = sources.flatten().tolist()
    target_points = targets.flatten().tolist()
    
    """
    # Read the input surface mesh
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()
    mesh = reader.GetOutput()
    """
    """
    # Read the input surface mesh from a PLY file
    reader = vtk.vtkPLYReader()
    reader.SetFileName(file_name)
    reader.Update()
    mesh = reader.GetOutput()
    """

    # Read the input surface mesh from a file
    file_extension = os.path.splitext(file_name)[1].lower()
    if file_extension == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif file_extension == ".ply":
        reader = vtk.vtkPLYReader()
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    
    reader.SetFileName(file_name)
    reader.Update()
    mesh = reader.GetOutput()

    # Extract centerline
    centerline_extractor = vmtkscripts.vmtkCenterlines()
    centerline_extractor.Surface = mesh

    # Define source and target points
    # http://www.vmtk.org/vmtkscripts/vmtkcenterlines.html
    centerline_extractor.SeedSelectorName = seedselector
    centerline_extractor.SourcePoints = source_points 
    centerline_extractor.TargetPoints = target_points

    centerline_extractor.Length = step_length     

    centerline_extractor.Execute()

    centerlines = centerline_extractor.Centerlines
    return centerlines

# The number of smoothing iterations, smoothing factor and resampling length are defalt with reference to 2021Biomed Eng Online Kjeldsberg (https://doi.org/10.1186/s12938-021-00957-6)
def vmtk_centerline_smoothing(centerlines, num_iteration=100, smooth_factor=1.2):
    #==============================================================================
                                                                                                                        
    # args: centerlines: centerlines (vtkPolydata)
    #       num_iteration: number of iteration (str)
    #       smooth_factor: smooth factor (float)
    # returns: smoothed centerline (vtkPolyData)
                                                                             
                                                                             
    #==============================================================================
    smoother = vmtkscripts.vmtkCenterlineSmoothing()
    smoother.Centerlines = centerlines
    smoother.NumberOfSmoothingIterations = num_iteration # smoother.NumberOfSmoothingIterations = num_iteration  # Set the number of smoothing iterations
    smoother.SmoothingFactor = smooth_factor # smoother.SmoothingFactor = smooth_factor  # Set the smoothing factor
    smoother.Execute()
    smoothed_centerlines = smoother.Centerlines
    return smoothed_centerlines

def vmtk_centerline_resampling(centerlines, step_length=0.1):
    #==============================================================================
                                                               
    # args: (post-smoothing) centerlines (vtkPolydata)
    #       step_length (float): resampling length
    # returns: resampled centerline (vtkPolyData)
    #==============================================================================
    resampler = vmtkscripts.vmtkCenterlineResampling()
    resampler.Centerlines = centerlines
    resampler.Length = step_length  # Set the resampling step length here
    resampler.Execute()
    resampled_centerline = resampler.Centerlines
    return resampled_centerline

# Compute centerline attributes and geometric features
# https://github.com/KVSlab/morphMan/blob/master/morphman/common/vmtk_wrapper.py
def vmtk_centerline_attributes(centerlines):
    #==============================================================================
    # args: centerlines (vtkPolydata)
    # returns: centerlines with attributes (vtkPolyData)
    #==============================================================================
    centerline_attributes = vmtkscripts.vmtkCenterlineAttributes()
    centerline_attributes.Centerlines = centerlines
    centerline_attributes.AbscissasArrayName = 'abscissas'
    centerline_attributes.NormalsArrayName = 'normals'
    centerline_attributes.Execute()
    centerlines = centerline_attributes.Centerlines
    return centerlines
def vmtk_compute_geometry(centerlines):
    #==============================================================================
    # args: centerlines (vtkPolyData)
    # returns: centerlines (vtkPolyData) with geometry
    #==============================================================================
    centerline_geometry = vmtkscripts.vmtkCenterlineGeometry()
    centerline_geometry.Centerlines = centerlines
    centerline_geometry.Execute()
    centerlines = centerline_geometry.Centerlines
    return centerlines


# vessel_data[point_coord] = {'radius': radius, "arc": arc, "curvature": curvature, "torsion": torsion}
# >> vessel_data[tuple(coord)]
def vmtk_vessel_imformation(centerlines):
    #==============================================================================
    # args: centerlines (vtkPolydata) post vmtk_centerline_attribute, vmtk_compute_geometry
    # returns: vessel infromation (dict) including radius (not diameter), arc length, curvature and torsion
    #==============================================================================     
    # Access the information of the vessels
    vessel_radius = centerlines.GetPointData().GetArray("MaximumInscribedSphereRadius")
    vessel_arc = centerlines.GetPointData().GetArray("abscissas")
    vessel_normal = centerlines.GetPointData().GetArray("normal") # >>None

    # Access the information of the centerline geometry
    vessel_fernettangent = centerlines.GetPointData().GetArray("FernetTangent") # >>None
    vessel_fernetnormal = centerlines.GetPointData().GetArray("FernetNormal") # >>None
    vessel_fernetbinormal = centerlines.GetPointData().GetArray("FernetBiNormal") # >>None
    vessel_curvature = centerlines.GetPointData().GetArray("Curvature")
    vessel_torsion = centerlines.GetPointData().GetArray("Torsion")
    vessel_tortuosity = centerlines.GetPointData().GetArray("Tortuosity") # >>Stored in cell data
    
    # vessel_data[point_coord] = {'radius': radius, "arc": arc, "curvature": curvature, "torsion": torsion}
    vessel_data = {}
    for i in range(vessel_radius.GetNumberOfTuples()):
        points_id = i
        point_coord = centerlines.GetPoint(i)
        radius = vessel_radius.GetTuple1(i)
        arc = vessel_arc.GetTuple1(i)
        curvature = vessel_curvature.GetTuple1(i)
        torsion = vessel_torsion.GetTuple1(i)    

        vessel_data[point_coord] = {
        "radius": radius,
        "arc": arc,
        "curvature": curvature,
        "torsion": torsion
        }
    return vessel_data

# Rmks: When resampled, the coordinates are different for each centerline, even for the same segment.
def vmtk_branch_splitting(centerlines):
    #==============================================================================
    # args: centerlines (vtkPolydata)
    # returns: splitted centerline (vtkPolydata) with information on blanking (blank; 1), CenterlineIds, TractIds, GroupIds, and Length and Tortuosity (if attribute and geometry information is available)
    #==============================================================================    
    branch_splitting = vmtkscripts.vmtkBranchExtractor()
    branch_splitting.Centerlines = centerlines
    
    """ # Renaming Radius to MaximumInscribedSphereRadius beforehand avoids errors
    # Check for 'Radius' or 'MaximumInscribedSphereRadius' in centerlines
    point_data = centerlines.GetPointData()
    array_names = [point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())]

    if 'Radius' in array_names:
        radiusArrayName = 'Radius'
    elif 'MaximumInscribedSphereRadius' in array_names:
        radiusArrayName = 'MaximumInscribedSphereRadius'
    else:
        raise ValueError("No appropriate radius array found in centerlines")
    branch_splitting.RadiusArrayName = radiusArrayName
    """

                                                                           
    branch_splitting.RadiusArrayName = 'MaximumInscribedSphereRadius'

    branch_splitting.Execute()
    splitted_centerlines = branch_splitting.Centerlines                                                        
    return splitted_centerlines

"""
def vmtk_branch_splitting(centerlines):
    #==============================================================================
    # args: centerlines (vtkPolydata)
    # returns: splitted centerline (vtkPolydata) with information on blanking (blank; 1), CenterlineIds, TractIds, GroupIds, and Length and Tortuosity (if attribute and geometry information is available)
    #==============================================================================    
    branch_splitting = vmtkscripts.vmtkBranchExtractor()
    branch_splitting.Centerlines = centerlines

    # Check for 'Radius' or 'MaximumInscribedSphereRadius' in centerlines
    point_data = centerlines.GetPointData()
    array_names = [point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())]

    if 'Radius' in array_names:
        radiusArrayName = 'Radius' # 3D Slicer uses Radius (also stores EdgeArray, EdgePCoordArray, SPACE)
    elif 'MaximumInscribedSphereRadius' in array_names:
        radiusArrayName = 'MaximumInscribedSphereRadius' # Python VMTK uses MaximumInscribedSphereRadius (also stores EdgeArray, EdgePCoordArray)
    else:
        raise ValueError("No appropriate radius array found in centerlines")

    branch_splitting.RadiusArrayName = radiusArrayName
    branch_splitting.Execute()
    splitted_centerlines = branch_splitting.Centerlines
    return splitted_centerlines # 3D Slicer stores: EdgeArray, EdgePCoordArray, Radius, Blanking, CenterlineIds, TractIds, GroupIds
"""

def access_cell_data(splitted_centerlines):
    #==============================================================================
    # args: splitted centerlines (post vmtk_branch_splitting) (vtkPolydata)
    # returns: cell_data including Blanking, CenterlineIds, Length, Tortuosity, TractIds and GroupIds
    #==============================================================================    
    cell_data = {}
    cellData = splitted_centerlines.GetCellData()
    # Access to each cell data array
    for i in range(cellData.GetNumberOfArrays()):
        array = cellData.GetArray(i)
        array_name = array.GetName()
        # List for storing array data
        data_list = []
        # Access the data in each cell
        for j in range(array.GetNumberOfTuples()):
            tuple_data = []
            for k in range(array.GetNumberOfComponents()):
                tuple_data.append(array.GetComponent(j, k))
            data_list.append(tuple_data)
        cell_data[array_name] = data_list
    return cell_data

def filter_specific_segment(splitted_centerlines, target_cell_number):
    #==============================================================================
    # args: splitted centerlines (post vmtk_branch_splitting) (vtkPolydata)
    # returns: segment with target_cell_number (vtkPolydata)
    #==============================================================================    
    cell_data = splitted_centerlines.GetCellData()
    group_ids = cell_data.GetArray('GroupIds')

    # Get information about "Blanking"
    blanking_array = splitted_centerlines.GetCellData().GetArray('Blanking')
    if blanking_array.GetValue(target_cell_number) == 0:
        # Initialize PolyData to store results
        specific_segment = vtk.vtkPolyData()
        specific_segment_points = vtk.vtkPoints()
        specific_segment_lines = vtk.vtkCellArray()

        for i in range(splitted_centerlines.GetNumberOfCells()):
            if group_ids.GetValue(i) == target_cell_number:
                cell = splitted_centerlines.GetCell(i)
                point_ids = cell.GetPointIds()
                specific_segment_lines.InsertNextCell(point_ids.GetNumberOfIds())
                for j in range(point_ids.GetNumberOfIds()):
                    point_id = point_ids.GetId(j)
                    point = splitted_centerlines.GetPoint(point_id)
                    new_point_id = specific_segment_points.InsertNextPoint(point)
                    specific_segment_lines.InsertCellPoint(new_point_id)

        specific_segment.SetPoints(specific_segment_points)
        specific_segment.SetLines(specific_segment_lines)
    
    elif blanking_array.GetValue(target_cell_number) == 1: 
        # not a segment, but bifurcation (Blanking)"
        specific_segment = None
    
    return specific_segment


# Rmks1: When resampled, Multiple centerlines exist in the same segment.
# Rmks2: In addition, the coordinates are different for each centerline, even for the same segment.
"""
def extract_coords_from_segment(segment):
    coords = []
    for i in range(segment.GetNumberOfPoints()):
        point = segment.GetPoint(i)
        coords.append(list(point))
    return np.array(coords)
"""
# Use the coordinates of cell number 0  as representative in each segment
# If cell number 0 is used, it can be used even when there is only one centerline.
def extract_coords_from_segment(segment, cell_number=0):
    #==============================================================================
    # args: segment (post filter_specific_segment with target_cell_number) (vtkPolydata)
    # returns: 3D coords(np.array) of cell No.0
    #==============================================================================  
    cell0 = segment.GetCell(cell_number)
    cell0_coords = []

    for i in range(cell0.GetNumberOfPoints()):
        point_id = cell0.GetPointId(i)
        coord = segment.GetPoint(point_id)
        cell0_coords.append(list(coord))

    return np.array(cell0_coords)

"""
def remove_ends(segment_coords, ratio=0.05):
    #==============================================================================
    # args: 3D coords (np.array)
    #       ratio: percentage to be removed based on total length
    # returns: 3D coords after deletion (np.array)
    #==============================================================================  
    diff_coords = np.diff(segment_coords, axis=0)
    ds = np.sqrt(np.sum(diff_coords**2, axis=1))
    segment_length = np.sum(ds)

    length_to_remove = ratio * segment_length
    # Remove from both ends of the curve by  ratio
    i = 0; j = 0
    while True:
        length_from_proximal_end = np.sum(ds[:i])
        if length_from_proximal_end < length_to_remove:
            i += 1
        else:
            break
    while True:
        length_from_distal_end = np.sum(ds[::-1][:j])
        if length_from_distal_end < length_to_remove:
            j += 1
        else:
            break
    return segment_coords[i:-j]  # since step length is constant, i ≈ j
"""

"""
def vmtk_centerlinemerge(splitted_centerlines):  
    #==============================================================================
    # args: splitted centerlines (post vmtk_branch_splitting) (vtkPolydata)
    # returns: merged centerline belonging to the same groups (vtkPolydata)
    #==============================================================================
    centerlinemerge = vmtkscripts.vmtkCenterlineMerge()
    centerlinemerge.Centerlines = splitted_centerlines
    centerlinemerge.Execute()
    merged_centerlines = centerlinemerge.Centerlines
    return merged_centerlines
"""

def vmtk_centerlinemerge(centerlines, step_length=0.1):  
    #==============================================================================
    # args: splitted centerlines (post vmtk_branch_splitting) (vtkPolydata)
    # returns: merged centerline belonging to the same groups (vtkPolydata)

    # Rmks: centerlinemerge includes resampling process
    #==============================================================================
    centerlinemerge = vmtkscripts.vmtkCenterlineMerge()
    centerlinemerge.Centerlines = centerlines
    centerlinemerge.Length = step_length     
    centerlinemerge.Execute()
    merged_centerlines = centerlinemerge.Centerlines
    return merged_centerlines 

"""
def vmtk_convert2numpy(centerlines):
    #==============================================================================
    # args: splitted and merged centerlines (post vmtk_branch_splitting and vmtk_centerlinemerge) (vtkPolydata)
    # returns: celldata: dict including 'Blanking', 'CenterlineIds', 'Length', 'Tortuosity', 'TractIds', 'GroupIds', and 'CellPointIds'
    #          pointids: list of sorted point IDs for segments ordered by "GroupIds"
    #          points: all coordinate in the centerlines ordered by "CellPointIds"
    #          pointdata: dict including 'Curvature', 'EdgeArray', 'EdgePCoordArray', 'FrenetBinormal', 'FrenetNormal', 'FrenetTangent', 'MaximumInscribedSphereRadius', 'Torsion', 'abscissas', 'normals', and 'TCoords' ordered by 'CellPointIds'
    #==============================================================================
    centerlines_to_numpy = vmtkcontribscripts.vmtkCenterlinesToNumpy()
    centerlines_to_numpy.Centerlines = centerlines
    centerlines_to_numpy.Execute()
    numpy_centerlines = centerlines_to_numpy.ArrayDict

    # numpy_centerlines includes 'CellData', 'Points', and 'PointData'
    celldata = numpy_centerlines.get('CellData') # celldata: 'Blanking', 'CenterlineIds', 'Length', 'Tortuosity', 'TractIds', 'GroupIds', 'CellPointIds'
    points = numpy_centerlines.get('Points') # points: coodinates ordered by 'CellPointIds'
    pointdata = numpy_centerlines.get('PointData') # pointdata: 'Curvature', 'EdgeArray', 'EdgePCoordArray', 'FrenetBinormal', 'FrenetNormal', 'FrenetTangent', 'MaximumInscribedSphereRadius', 'Torsion', 'abscissas', 'normals', and 'TCoords' ordered by 'CellPointIds'
    
    pointids = celldata["CellPointIds"]

    return celldata, pointids, points, pointdata
"""

                                                   
class SilentCenterlinesToNumpy(vmtkcontribscripts.vmtkCenterlinesToNumpy):
    def PrintLog(self, message):
                          
        pass
def vmtk_convert2numpy(centerlines):
    centerlines_to_numpy = SilentCenterlinesToNumpy()              
    centerlines_to_numpy.Centerlines = centerlines
    centerlines_to_numpy.Execute()
    numpy_centerlines = centerlines_to_numpy.ArrayDict

    celldata = numpy_centerlines.get('CellData')
    points = numpy_centerlines.get('Points')
    pointdata = numpy_centerlines.get('PointData')
    pointids = celldata["CellPointIds"]

    return celldata, pointids, points, pointdata


def remove_ends(pointids, ratio=0.05):
    #==============================================================================
    # args: list of sorted point IDs for segments (Rmks: Resampling process has already been completed.)
    #       ratio: percentage to be removed based on total length
    # returns: list of point IDs for segments after removing a specified number of elements from the start and end
    #==============================================================================  
    new_list_pointids = []

    for ids in pointids:
        #  calculate the number of elements in the ratio and ceil to the nearest integer
        remove_count = math.ceil(ratio * len(ids))

        # add an empty array if the number of elements to be deleted is more than half of the original array
        if remove_count >= len(ids) // 2:
            new_list_pointids.append(np.array([]))
            continue
    
        # remove a specified number of elements from the start and end
        trimmed = ids[remove_count:-remove_count]
        new_list_pointids.append(trimmed)
    return new_list_pointids

                              
def get_segment_coords(vtkPolyData):
    points = vtkPolyData.GetPoints()  # Get the vtkPoints object
    segment_num = vtkPolyData.GetNumberOfLines()
    
    segment_list = []

    # Loop for each segment
    for i in range(segment_num):
        segment_coords = []
        # Loop for coord ids in each segment
        for j in range (vtkPolyData.GetCell(i).GetNumberOfPoints()):
            pointid = vtkPolyData.GetCell(i).GetPointId(j)
            point = list(vtkPolyData.GetPoint(pointid))
            segment_coords.append(point)

        segment_list.append(segment_coords)

    return segment_list

from itertools import groupby
def segment_sort(vtkPolyData):
    segment_list = get_segment_coords(vtkPolyData)
    
                  
    segment_list.sort(key=lambda x: x[0])

                                                                         
    grouped_list = [list(g) for k, g in groupby(segment_list, key=lambda x: x[0])]

    return grouped_list

def check_steplength(centerlines):
    plt.figure()
    for segs in segment_sort(centerlines):
        for i in range(len(segs)):
            diff = np.sqrt(np.sum(np.diff(segs[i], axis=0)**2, axis=1))
            plt.hist(diff, bins=1000, range=(0,0.5), alpha=0.5)
    plt.xlabel("step length")
    plt.ylabel("number")
    plt.show()



# https://github.com/ajgeers/utils/blob/master/utils/vmtklib.py
# https://github.com/vmtk/vmtk/blob/master/vmtkScripts/vmtkscripts.py

from vmtk import vmtkscripts
def vmtksurfacecurvature(surface, curvature_type='mean', absolute_curvature=0,
                         median_filtering=0):
    """Compute curvature of an input surface.

    Args:
        surface: Surface mesh.
        curvature_type ('mean', 'gaussian', 'maximum', 'minimum'): Type of
            curvature to compute.
        absolute_curvature (bool): Output the absolute value of curvature.
        median_filtering (bool): Output curvature after median filtering to
            suppress numerical noise speckles.

    Returns:
        Surface with curvature variable.
        Pointdata:
            Curvature: Local surface curvature.

    """
    curvaturefilter = vmtkscripts.vmtkSurfaceCurvature()
    curvaturefilter.Surface = surface
    curvaturefilter.CurvatureType = curvature_type
    curvaturefilter.AbsoluteCurvature = absolute_curvature
    curvaturefilter.MedianFiltering = median_filtering
    curvaturefilter.Execute()
    return curvaturefilter.Surface


# https://github.com/ajgeers/utils/blob/master/utils/vmtklib.py
def vmtksurfacedecimation(surface, target=.9):
    """Reduce the number of triangles in a surface.

    Args:
        surface: Surface mesh.
        target: Desired number of triangles relative to the input number of
            triangles.

    Returns:
        Surface mesh with fewer triangles.

    """
    decimator = vmtkscripts.vmtkSurfaceDecimation()
    decimator.Surface = surface
    decimator.TargetReduction = target
    decimator.Execute()
    return decimator.Surface

from vtkmodules.util import numpy_support

def compute_curvatures(polydata):
                    
    curvaturesFilter = vtk.vtkCurvatures()
    curvaturesFilter.SetInputData(polydata)

             
    curvaturesFilter.SetCurvatureTypeToMean()
    curvaturesFilter.Update()
    mean_curvatures = numpy_support.vtk_to_numpy(curvaturesFilter.GetOutput().GetPointData().GetScalars())

              
    curvaturesFilter.SetCurvatureTypeToGaussian()
    curvaturesFilter.Update()
    gaussian_curvatures = numpy_support.vtk_to_numpy(curvaturesFilter.GetOutput().GetPointData().GetScalars())


                                     
    return gaussian_curvatures, mean_curvatures


import json
# Load inlet/outlet control points from a 3S Slicer .mrk.json file and return them as an (N, 3) NumPy array
                                                                           
def load_markups_positions(path):
    with open(path, 'r') as f:
        data = json.load(f)
    markups = data.get('markups', [])
    if not markups:
        return np.empty((0, 3))
    cps = markups[0].get('controlPoints', [])
    return np.array([cp['position'] for cp in cps], dtype=float)