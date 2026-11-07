#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: qc mesh alignment."""

########################################################
# ===== USER SETTINGS =====
                      
                               
                                                   
                                                                        
target_directory_ids = None                                                                         

# ===== Visualization settings =====
plotter_window_size = (2560, 1440)                            
plotter_position = (500, 500)                   
########################################################

import os
import sys
import json
import numpy as np
import pandas as pd
import pyvista as pv

# Add the path to make the utils module importable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
utils_path = os.path.join(project_root, 'lib')
mesh_path = os.path.join(project_root, '03_mesh')
sys.path.insert(0, os.path.join(project_root, 'lib'))
sys.path.append(utils_path)
sys.path.append(mesh_path)

from lib import file_utils as utils
import mesh_generation
import centerline_extraction

print("="*80)
print("Alignment check for brain_mesh, vessel_mesh, and merged_centerlines")
print("="*80)

                     
participants_csv_path = os.path.join(project_root, 'results', 'participants', 'IXI_HH_participants.csv')
if not os.path.exists(participants_csv_path):
    print(f"Error: CSV file not found: {participants_csv_path}")
    sys.exit(1)

df_participants = pd.read_csv(participants_csv_path)
all_directory_ids = df_participants['directory_id'].dropna().unique().tolist()

                      
if target_directory_ids is None:
    directory_ids = sorted(all_directory_ids)
else:
    if isinstance(target_directory_ids, str):
        target_directory_ids = [target_directory_ids]
    directory_ids = [did for did in sorted(all_directory_ids) if did in target_directory_ids]

print(f"directory_ids to process: {len(directory_ids)}")
if len(directory_ids) > 0:
    print(f"directory_id list: {directory_ids[:10]}{'...' if len(directory_ids) > 10 else ''}")
else:
    print("Warning: no directory_ids to process.")
    sys.exit(1)
print("="*80)

def check_alignment(directory_id):
    print(f"\n{'='*80}")
    print(f"Checking: {directory_id}")
    print(f"{'='*80}")
    
    missing_files = []
    
               
    vessel_mesh_dir = os.path.join(project_root, 'data', 'preprocessed', 'MRA_preprocessed', 'vessel_meshes', directory_id)
    vessel_mesh_path = os.path.join(vessel_mesh_dir, f"{directory_id}_vessel_mesh.ply")
    
    vessel_centerlines_dir = os.path.join(project_root, 'data', 'derived', 'vessel_centerlines', directory_id)
    merged_centerlines_path = os.path.join(vessel_centerlines_dir, f"{directory_id}_merged_centerlines.vtp")
    
    brain_mesh_dir = os.path.join(project_root, 'data', 'preprocessed', 'T1_preprocessed', 'brain_meshes', directory_id)
    brain_mesh_lt_path = os.path.join(brain_mesh_dir, f"{directory_id}_lt_pial_remesh.ply")
    brain_mesh_rt_path = os.path.join(brain_mesh_dir, f"{directory_id}_rt_pial_remesh.ply")
    
               
    if not os.path.exists(vessel_mesh_path):
        print(f"  Error: vessel_mesh not found: {vessel_mesh_path}")
        missing_files.append('vessel_mesh')
        return False, missing_files
    
    if not os.path.exists(merged_centerlines_path):
        print(f"  Error: merged_centerlines not found: {merged_centerlines_path}")
        missing_files.append('merged_centerlines')
        return False, missing_files
    
    if not os.path.exists(brain_mesh_lt_path):
        print(f"  Warning: brain_mesh_lt not found: {brain_mesh_lt_path}")
        missing_files.append('brain_mesh_lt')
    if not os.path.exists(brain_mesh_rt_path):
        print(f"  Warning: brain_mesh_rt not found: {brain_mesh_rt_path}")
        missing_files.append('brain_mesh_rt')
    
                          
    try:
        vessel_mesh = mesh_generation.load_ply_to_polydata(vessel_mesh_path)
        merged_centerlines = centerline_extraction.load_vtp(merged_centerlines_path)
        
        brain_mesh_lt = None
        if os.path.exists(brain_mesh_lt_path):
            brain_mesh_lt = pv.read(brain_mesh_lt_path)
        brain_mesh_rt = None
        if os.path.exists(brain_mesh_rt_path):
            brain_mesh_rt = pv.read(brain_mesh_rt_path)
        
                          
        celldata, pointids, points, pointdata = centerline_extraction.vmtk_convert2numpy(merged_centerlines)
        
                     
        pl = pv.Plotter(shape=(1, 2), window_size=plotter_window_size)
        
                            
        pl.subplot(0, 0)
        pl.add_mesh(vessel_mesh, color='lightblue', opacity=0.5)
        pl.add_mesh(merged_centerlines, color='lime', line_width=3)
        
        if brain_mesh_lt is not None:
            pl.add_mesh(brain_mesh_lt, color="#fafae1", opacity=1)

        pl.camera_position = 'xy'
        pl.camera.azimuth = 180
        pl.camera.zoom(1.5)
        pl.add_title(f"{directory_id} - Front View", font_size=16)
        
                             
        pl.subplot(0, 1)
        pl.add_mesh(vessel_mesh, color='lightblue', opacity=0.5)
        pl.add_mesh(merged_centerlines, color='lime', line_width=3)
        if brain_mesh_lt is not None:
            pl.add_mesh(brain_mesh_lt, color="#fafae1", opacity=1)
        
        pl.camera_position = 'xy'
        pl.camera.azimuth = 270
        pl.camera.zoom(1.5)
        pl.add_title(f"{directory_id} - Left lateral View", font_size=16)
        
                                                              
        pl.render()
        if hasattr(pl, 'render_window') and pl.render_window is not None:
            pl.render_window.SetPosition(plotter_position[0], plotter_position[1])
        
                      
        def close_on_space():
            pl.close()
        
        pl.add_key_event('space', close_on_space)
        pl.add_text("Press SPACE to close and continue", position='lower_left', font_size=14, color='red')
        print(f"  Showing visualization window... (press SPACE to continue)")
        
        try:
            pl.show(interactive=True)
        except (AttributeError, RuntimeError):
                                                                    
            pass
        
        return True, missing_files
        
    except Exception as e:
        print(f"  Error: failed while loading or visualizing mesh: {e}")
        import traceback
        traceback.print_exc()
        return False, missing_files

                  
total_count = 0
success_count = 0
error_count = 0
all_missing_files = {}

for directory_id in directory_ids:
    total_count += 1
    try:
        success, missing_files = check_alignment(directory_id)
        if success:
            success_count += 1
        else:
            error_count += 1
        
        if missing_files:
            all_missing_files[directory_id] = missing_files
            
    except KeyboardInterrupt:
        print(f"\n\nProcessing interrupted.")
        print(f"Current position: {directory_id}")
        print(f"To resume: set target_directory_ids starting from '{directory_id}'")
        break
    except Exception as e:
        error_count += 1
        print(f"\n  Error: failed while processing {directory_id}: {e}")
        import traceback
        traceback.print_exc()

         
print("\n" + "="*80)
print("Processing complete")
print("="*80)
print(f"Processed directory_id count: {total_count}")
print(f"Succeeded: {success_count}")
print(f"Errors: {error_count}")

if all_missing_files:
    print(f"\ndirectory_ids with missing files: {len(all_missing_files)}")
    print("Missing file details:")
    for dir_id, missing in all_missing_files.items():
        print(f"  {dir_id}: {', '.join(missing)}")
else:
    print("\nAll files were present.")

print("="*80)
