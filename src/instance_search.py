import numpy as np 
import cv2 
import pandas as pd
import glob # for file type
import time 

# Path of downloaded datasets
download_path='/Users/txsing/datasets_4186' 
# Path of query images
path_query=download_path+'/query_4186'
# Path of query images' bounding box
path_query_txt=download_path+'/query_txt_4186'
# Path of the gallery
path_gallery=download_path+'/gallery_4186'

name_query=glob.glob(path_query+'/*.jpg')
num_query=len(name_query)
name_gallery=glob.glob(path_gallery+'/*.jpg')
num_gallery=len(name_gallery)