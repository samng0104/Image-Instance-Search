import numpy as np
import cv2
import glob  # for file type
import time

import torch
import torchvision.models as models



def query_crop(query_img, txt, save_path):
    # query_img = cv2.imread(query_path)
    query_img = query_img[:, :, ::-1]  # bgr2rgb
    crop = query_img[int(txt[1]):int(txt[1] + txt[3]), int(txt[0]):int(txt[0] + txt[2]), :]  # crop the instance region
    cv2.imwrite(save_path, crop[:, :, ::-1])  # save the cropped region
    return crop

download_path = 'C:\\Users\\lungpng2\\Documents\\datasets_4186' # Path of downloaded datasets
path_query = download_path+'\\query_4186' # Path of query images: '\\Users\\samng\\Documents\\datasets_4186\\query_4186'
path_query_txt = download_path+'\\query_txt_4186' # Path of query images' bounding box '\\Users\\samng\\Documents\\datasets_4186\\query_txt_4186'
path_gallery = download_path+'\\gallery_4186' # Path of the gallery '\\Users\\samng\\Documents\\datasets_4186\\gallery_4186'
save_path = download_path + '\\cropped_4186' # Path of cropped image

name_query = glob.glob(path_query+'\\*.jpg') # File name of query images
num_query = len(name_query) # Total number of query images
name_gallery = glob.glob(path_gallery+'\\*.jpg') # File name of gallery images
num_gallery = len(name_gallery) # Total number of gallery images

# ORB detector as SIFT detector
sift = cv2.ORB_create()
record_all = np.zeros((num_query, len(name_gallery))) # record for similarity, initialize to 0 array
query_imgs_no = [x.split('\\')[-1][:-4] for x in glob.glob(path_query+'\\*.jpg')] # Image number of query images (eg. 2714, 776)
gallery_imgs_no = [x.split('\\')[-1][:-4] for x in glob.glob(path_gallery+'\\*.jpg')] # Image number of gallery images (eg. 448)

"""
ResNet101 Feature Extraction
"""
def ResNet101_extraction()


# Output
f = open(r'./rank_list.txt', 'w')
for i in range(num_query):
    f.write('Q'+str(i+1)+': ')
    for j in range(len(name_gallery)):
        f.write(str(np.int32(record_all[i, j]))+' ')
    f.write('\n')
f.close()
