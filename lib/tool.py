import time
import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.pagesizes import A4

def measure_time(func, *args, **kwargs):
    start = time.time()                 
    result = func(*args, **kwargs)              
    end = time.time()                 
    time_diff = end - start                

                  
    hours, remainder = divmod(time_diff, 3600)
    minutes, seconds = divmod(remainder, 60)

                  
    print("処理時間: {:02}時間{:02}分{:02}秒".format(int(hours), int(minutes), int(seconds)))
    return result                
    

def notify(case):
    os.system('afplay /System/Library/Sounds/Hero.aiff')
    finish_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Finish, {case} at {finish_time}!")

    
def remove_outliers_10(data):
                            
    percentile_10 = np.percentile(data, 10)
    percentile_90 = np.percentile(data, 90)
    print(f'10th percentile: {percentile_10:.5f}')
    print(f'90th percentile: {percentile_90:.5f}')
    
                                   
    filtered_data = np.clip(data, percentile_10, percentile_90)

                   
    count = np.sum((data <= percentile_10) | (data >= percentile_90))
    percentage = (count / len(data)) * 100
    print(f'Percentage of outliers: {percentage:.5f}%')

    return filtered_data

def remove_outliers_iqr(data):
                              
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
                 
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f'Lower bound (Q1 - 1.5 * IQR): {lower_bound:.5f}')
    print(f'Upper bound (Q3 + 1.5 * IQR): {upper_bound:.5f}')
    
             
    filtered_data = np.clip(data, lower_bound, upper_bound)
    
                   
    count = np.sum((data < lower_bound) | (data > upper_bound))
    percentage = (count / len(data)) * 100
    print(f'Percentage of outliers: {percentage:.5f}%')

    return filtered_data
    
"""
def remove_outliers_iqr(data):
    # 第1四分位数（Q1）と第3四分位数（Q3）を計算
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    # 外れ値のしきい値を計算
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f'Lower bound: {lower_bound:.5f}')
    print(f'Upper bound: {upper_bound:.5f}')
    
    # 外れ値を除去
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    # 除去されたものの割合を計算
    count = np.sum((data < lower_bound) | (data > upper_bound))
    percentage = (count / len(data)) * 100
    print(f'Percentage of outliers: {percentage:.5f}%')

    return filtered_data
"""


def remove_end_points(pointids, removal):
    result_pointids = []
    
    for id_group in pointids:
        if len(id_group) <= 2 * removal:
            continue
        else:
            new_id_group = id_group[removal:-removal]
            result_pointids.append(new_id_group)
    

    # print("percentage of removed points: ", (1 - len(result_pointids) / len(pointids)) * 100, "%")
    return result_pointids

def remove_end_points2(pointids, removal, 
                      exclusion_criteria=5,                                      
                      s_length = 0.2, 
                      voxel_size = 0.5):
    
    result_pointids = []
    for id_group in pointids:
        if len(id_group) <= 2 * removal + int(exclusion_criteria / (s_length * voxel_size)):                               
            continue
        else:
            new_id_group = id_group[removal:-removal]            
            result_pointids.append(new_id_group)
    return result_pointids

def lines_from_points(points):
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points) - 1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points) - 1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly


                              
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

import time

def measure_time(func, *args, **kwargs):
    start = time.time()                 
    result = func(*args, **kwargs)              
    end = time.time()                 
    time_diff = end - start                

                  
    hours, remainder = divmod(time_diff, 3600)
    minutes, seconds = divmod(remainder, 60)

                  
    print("処理時間: {:02}時間{:02}分{:02}秒".format(int(hours), int(minutes), int(seconds)))
    return result                

def lines_from_points(points):
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points) - 1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points) - 1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly

import ants
import matplotlib.animation as animation
def create_animation(slice_count, ants_T1, ants_MRA, filename):
    mi = ants.image_mutual_information(ants_T1, ants_MRA)
    img_T1 = ants_T1.numpy()
    img_MRA = ants_MRA.numpy()
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

             
    ims = []
    for z in range(slice_count):
        im0 = ax[0].imshow(img_T1[:, z, :])
        im1 = ax[0].imshow(img_MRA[:, z, :], cmap='gray', alpha=0.5)
        im2 = ax[1].imshow(img_T1[:, z, :])
        im3 = ax[2].imshow(img_MRA[:, z, :], cmap='gray')
        ax[0].set_title(f"mutual_information={mi}")
        ims.append([im0, im1, im2, im3])

               
    ani = animation.ArtistAnimation(fig, ims, interval=300, repeat_delay=1000)
    ani.save(filename)

    plt.show()


import os
def create_dir(path):
                                        
    if not os.path.exists(path):
                         
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")


def save_result_as_pdf(result_text, output_path):
    pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5'))

    styles = getSampleStyleSheet()
    style = styles["Normal"]
    style.fontName = "HeiseiKakuGo-W5"
    style.fontSize = 20
    style.leading = 28

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    story = []

    for line in result_text.split('\n'):
        if line.strip():
            story.append(Paragraph(line, style))
        else:
            story.append(Paragraph('<br/>', style))

    doc.build(story)
    print(f"PDF saved to: {output_path}")


def close_window(pl=None):
                                
    if pl is not None:
        try:
            pl.close()
        except:
            pass


def setup_matplotlib_close_on_space(fig):
                                       
    def on_key(event):
        if event.key == ' ':
            plt.close(fig)
    fig.canvas.mpl_connect('key_press_event', on_key)



