#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Public pipeline module: qc registration mmi."""

"""
=> Existing metrics (e.g. MMI) are not useful
due to an extreme imbalance in information density
T1 (dense): meaningful signal fills ~100% of space (parenchyma, WM, GM, ventricles, scalp, etc.).
MRA (sparse): >95% is empty/background noise; meaningful signal (vessels) is tiny.
MMI would compare T1 GM intensity vs MRA background; MRA is mostly zero, so T1 changes stay uncorrelated (score ~0).
Background (99%) uncorrelation swamps vessel (1%) correlation, so MMI fails as a metric.
"""

########################################################
# ===== USER SETTINGS =====
directory_id = "IXI013-HH-1212"
# directory_id = "IXI000-HH-0000"
########################################################

import os
import sys
import ants
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nibabel as nib
import pyvista as pv

# Add the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

            
csv_path = os.path.join(project_root, 'results', 'participants', 'IXI_HH_participants.csv')

          
df = pd.read_csv(csv_path)

                  
all_directory_ids = df['directory_id'].dropna().unique()
                      
directory_ids = sorted(all_directory_ids.tolist())

print(f"Extracted directory_id count: {len(all_directory_ids)}")
print(f"directory_ids to process: {len(directory_ids)}")


# 1. T1 (CAT12 processed)
post_cat12_T1_dir = os.path.join(project_root, 'data', 'preprocessed', 'T1_preprocessed', 'post_cat12', directory_id+"_cat12", "mri")
preprocessed_img_T1_path = os.path.join(post_cat12_T1_dir, 'm1_1_resampled_T1.nii')
segmented_brain_T1_path = os.path.join(post_cat12_T1_dir, 'p01_1_resampled_T1.nii')

# 2. MRA (Registered)
temporary_MRA_dir = os.path.join(project_root, 'data', 'preprocessed', 'MRA_preprocessed', 'temporary', directory_id)
TRSAA_SyNOnly_moving_img_MRA_path = os.path.join(temporary_MRA_dir, f'{directory_id}_MRA_TRSAA_SyNOnly_registered.nii.gz')
mapped_TRSAA_SyNOnly_bet_mask_MRA_path = os.path.join(temporary_MRA_dir, f'{directory_id}_MRA_TRSAA_SyNOnly_mapped_bet_mask.nii.gz')

          
required_files = [preprocessed_img_T1_path, segmented_brain_T1_path, TRSAA_SyNOnly_moving_img_MRA_path, mapped_TRSAA_SyNOnly_bet_mask_MRA_path]
if not all(os.path.exists(f) for f in required_files):
    print(f"Skipping {directory_id}: Files missing.")

        
preprocessed_img_T1 = ants.image_read(preprocessed_img_T1_path)
TRSAA_SyNOnly_moving_img_MRA = ants.image_read(TRSAA_SyNOnly_moving_img_MRA_path)

            
segmented_brain_T1 = ants.image_read(segmented_brain_T1_path)
fixed_mask = ants.get_mask(segmented_brain_T1)          

mapped_TRSAA_SyNOnly_bet_mask_MRA = ants.image_read(mapped_TRSAA_SyNOnly_bet_mask_MRA_path)                    

                               
                       
                                 
mask_intersection = fixed_mask * mapped_TRSAA_SyNOnly_bet_mask_MRA

               
if mask_intersection.sum() == 0:
    print(f"Warning {directory_id}: No overlap found.")

                              
                          
                                                           
score = ants.image_similarity(
    fixed_image=preprocessed_img_T1,
    moving_image=TRSAA_SyNOnly_moving_img_MRA,
    metric_type='MattesMutualInformation',
    fixed_mask=mask_intersection,
    moving_mask=mask_intersection,
    sampling_percentage=1.0
)
    
print(score)

print(preprocessed_img_T1)
print(TRSAA_SyNOnly_moving_img_MRA)

ants.plot(preprocessed_img_T1, TRSAA_SyNOnly_moving_img_MRA)
ants.plot(preprocessed_img_T1, fixed_mask)
ants.plot(TRSAA_SyNOnly_moving_img_MRA, mapped_TRSAA_SyNOnly_bet_mask_MRA)
ants.plot(preprocessed_img_T1, mapped_TRSAA_SyNOnly_bet_mask_MRA)
