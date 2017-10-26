import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import os
from tqdm import tqdm 
import sys


def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return(f)


setName=sys.argv[1]

luna_path = "/new_disk_1/tianchi/"
luna_subset_path = luna_path+("data/%s_set/" % setName)
output_path = ("/new_disk_1/tianchi/data/lfz_data_%s/" % setName)
file_list=glob(luna_subset_path+"*.mhd")

# The locations of the nodes
df_node = pd.read_csv(luna_path+'data/csv/'+setName+'/annotations.csv')
df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
df_node = df_node.dropna()

chunk_size = 64

for fcount, img_file in enumerate(tqdm(file_list)):
  mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file

  xx = img_file.split('/')[-1]
  yy  = xx.split('.')[0]


     
  if mini_df.shape[0]>0: # some files may not have a nodule--skipping those 
    itk_img = sitk.ReadImage(img_file) 
    origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
       
  file = img_file.split('/')[-1]

  label = []
  for node_idx, cur_row in mini_df.iterrows():       
    node_x = cur_row["coordX"]
    node_y = cur_row["coordY"]
    node_z = cur_row["coordZ"]
    diam = cur_row["diameter_mm"]

            
    center = np.array([node_x, node_y, node_z])   # nodule center
    v_center = np.rint((center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)
             
    v_center = abs(v_center)
    label.append([v_center[2],v_center[1],v_center[0],diam])
  print label,file
  name  = file.split('.')[0]
  np.save(output_path+name+'_label.npy',np.array(label))

