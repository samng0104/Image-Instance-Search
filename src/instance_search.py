import numpy as np 
import cv2 
import pandas as pd
import glob # for file type
import time 

# Path of downloaded datasets
download_path='C:\Users\Sam NG\Downloads\datasets_4186'
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

# ORB detector as SIFT detector
sift = cv2.ORB_create()
# record for similarity
record_all=np.zeros((num_query,len(name_gallery)))

query_imgs_no = [x.split('/')[-1][:-4] for x in glob.glob(path_query+'/*.jpg')]
gallery_imgs_no = [x.split('/')[-1][:-4] for x in glob.glob(path_gallery+'/*.jpg')]

# Method 1: Bag of Visual Words

# Method 2: Convolutional Neural Networks 