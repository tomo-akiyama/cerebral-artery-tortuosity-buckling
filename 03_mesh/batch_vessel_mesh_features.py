#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: batch vessel mesh features."""

import pandas as pd
import numpy as np
import os
import sys

              
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

            
csv_path = os.path.join(project_root, 'results', 'participants', 'IXI_HH_participants.csv')

          
df = pd.read_csv(csv_path)

                  
directory_ids = df['directory_id'].dropna().unique()
print(f"抽出されたdirectory_id数: {len(directory_ids)}")
print(f"directory_id一覧: {directory_ids.tolist()}")

                
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

                               
utils_path = os.path.join(project_root, 'lib')
sys.path.insert(0, os.path.join(project_root, 'lib'))  # Add analysis_code to path for package imports
sys.path.append(utils_path)
                                                                 

from run_vessel_mesh_features import calculate_and_save_vessel_mesh_features
import utils
import mesh_generation
import centerline_extraction

    
voxel_size = 0.5  # mm

                        
results = []
failed_cases = []

for idx, directory_id in enumerate(directory_ids, 1):
    print(f"\n{'='*60}")
    print(f"処理中 ({idx}/{len(directory_ids)}): {directory_id}")
    print(f"{'='*60}")
    
    try:
                    
        case_id = directory_id[3:6].lstrip("0")
        if case_id == "":
            case_id = "0"
        case_id = int(case_id)
        
        print(f"directory_id: {directory_id}")
        print(f"case_id: {case_id}")
        
               
        vessel_mesh_dir = os.path.join(
            project_root,
            'data',
            'preprocessed',
            'MRA_preprocessed',
            'vessel_meshes',
            directory_id,
        )
        vessel_vtk_mc_taubin_butterfly_path = os.path.join(
            vessel_mesh_dir,
            f"{directory_id}_vessel_mesh.ply"
        )
        merged_splitted_centerlines_path = os.path.join(
            project_root,
            'data',
            'derived',
            'vessel_centerlines',
            directory_id,
            f"{directory_id}_merged_centerlines.vtp"
        )
        
                                        
        required_files = [vessel_vtk_mc_taubin_butterfly_path, merged_splitted_centerlines_path]
        if not utils.ensure_files_available(required_files, max_wait_time=120):
            print("警告: 必要なファイルが利用できない")
            failed_cases.append({"directory_id": directory_id, "reason": "必要なファイルがオンラインのみで、ダウンロードできなかった"})
            continue
        
                         
        if not os.path.exists(vessel_vtk_mc_taubin_butterfly_path):
            print(f"警告: ファイルが見つからない: {vessel_vtk_mc_taubin_butterfly_path}")
            failed_cases.append({"directory_id": directory_id, "reason": f"ファイルが見つからない: {vessel_vtk_mc_taubin_butterfly_path}"})
            continue
            
        if not os.path.exists(merged_splitted_centerlines_path):
            print(f"警告: ファイルが見つからない: {merged_splitted_centerlines_path}")
            failed_cases.append({"directory_id": directory_id, "reason": f"ファイルが見つからない: {merged_splitted_centerlines_path}"})
            continue
        
                  
        vessel_vtk_mc_taubin_butterfly = mesh_generation.load_ply_to_polydata(vessel_vtk_mc_taubin_butterfly_path)
        merged_splitted_centerlines = centerline_extraction.load_vtp(merged_splitted_centerlines_path)
        
                                                 
        df_vessel_mesh_features = calculate_and_save_vessel_mesh_features(
            vessel_mesh_vtk=vessel_vtk_mc_taubin_butterfly,
            vessel_centerlines_vtk=merged_splitted_centerlines,
            voxel_size=voxel_size,
            directory_id=directory_id,
            csv_path=None,
            output_dir=vessel_mesh_dir
        )
        
        results.append({"directory_id": directory_id, "status": "success", "data": df_vessel_mesh_features})
        print(f"✓ 完了: {directory_id}")
        
    except Exception as e:
        print(f"✗ エラー: {directory_id}")
        print(f"  エラー内容: {str(e)}")
        import traceback
        traceback.print_exc()
        failed_cases.append({"directory_id": directory_id, "reason": str(e)})

         
print(f"\n{'='*60}")
print("処理完了")
print(f"{'='*60}")
print(f"成功: {len(results)}件")
print(f"失敗: {len(failed_cases)}件")

if failed_cases:
    print("\n失敗したケース:")
    for fc in failed_cases:
        print(f"  - {fc['directory_id']}: {fc['reason']}")

print(f"\n結果は各ケースの vessel_meshes/{{id}}/{{id}}_vessel_mesh_features.csv に保存した")

