"""
Method 1: Bag of Visual Word
"""
import numpy as np
import cv2
import pandas as pd
import glob  # for file type
import time

# For Bag of Words implementation
from sklearn.cluster import KMeans


def query_crop(query_img, txt, save_path):
    # query_img = cv2.imread(query_path)
    query_img = query_img[:, :, ::-1]  # bgr2rgb
    crop = query_img[int(txt[1]):int(txt[1] + txt[3]), int(txt[0]):int(txt[0] + txt[2]), :]  # crop the instance region
    cv2.imwrite(save_path, crop[:, :, ::-1])  # save the cropped region
    return crop

# Assigns visual words to the descriptors using KMeans clustering.
def get_visual_words(descriptor_list, kmeans):
    # Stack all the descriptors vertically
    descriptors = np.vstack(descriptor_list)
    # Perform KMeans clustering to get visual words
    kmeans.fit(descriptors)
    # Return the visual words (cluster centroids)
    return kmeans.cluster_centers_

# Extracts SIFT descriptors from an image.
def get_image_descriptor(image, sift):
    kp, des = sift.detectAndCompute(image, None)
    return des

download_path = '/Users/samng/Documents/datasets_4186' # Path of downloaded datasets
path_query = download_path+'/query_4186' # Path of query images: '/Users/samng/Documents/datasets_4186/query_4186'
path_query_txt = download_path+'/query_txt_4186' # Path of query images' bounding box '/Users/samng/Documents/datasets_4186/query_txt_4186'
path_gallery = download_path+'/gallery_4186' # Path of the gallery '/Users/samng/Documents/datasets_4186/gallery_4186'
save_path = download_path + '/cropped_4186' # Path of cropped image

name_query = glob.glob(path_query+'/*.jpg') # File name of query images
num_query = len(name_query) # Total number of query images
name_gallery = glob.glob(path_gallery+'/*.jpg') # File name of gallery images
num_gallery = len(name_gallery) # Total number of gallery images

# ORB detector as SIFT detector
sift = cv2.ORB_create()
record_all = np.zeros((num_query, len(name_gallery))) # record for similarity, initialize to 0 array
query_imgs_no = [x.split('/')[-1][:-4] for x in glob.glob(path_query+'/*.jpg')] # Image number of query images (eg. 2714, 776)
gallery_imgs_no = [x.split('/')[-1][:-4] for x in glob.glob(path_gallery+'/*.jpg')] # Image number of gallery images (eg. 448)

# Default Handcraft method: SIFT
# the iteration loop for query
for i, query_img_no in enumerate(query_imgs_no[0:1]):
    time_s = time.time()
    dist_record = []

    per_query_name = path_query + '/' + str(query_img_no) + '.jpg' # /Users/samng/Documents/datasets_4186/query_4186/2714.jpg
    per_query = cv2.imread(per_query_name) # per_query is an image
    txt_path = path_query_txt + '/' + str(query_img_no) + '.txt' # /Users/samng/Documents/datasets_4186/query_txt_4186/2714.txt
    per_query = per_query[:, :, ::-1]
    txt = np.loadtxt(txt_path)
    # [244.  49. 262. 252.]
    per_save_path = save_path + '/' + str(query_img_no) + '.jpg'
    # /Users/samng/Documents/datasets_4186/cropped_4186/2714.jpg
    
    # feature extraction for per query
    per_query = query_crop(per_query, txt, per_save_path)
    # quite naive, just an example
    per_query_kp, per_query_des = sift.detectAndCompute(per_query, None)

    # the iteration loop for gallery
    for j, gallery_img_no in enumerate(gallery_imgs_no):
        per_gallery_name = path_gallery+'/'+str(gallery_img_no)+'.jpg'
        per_gallery = cv2.imread(per_gallery_name)
        # feature extraction for per gallery
        per_gallery_kp, per_gallery_des = sift.detectAndCompute(per_gallery, None)
        # use part of the features to make the calculation feasible
        # quite naive, just an example
        min_kp_num = np.amin([len(per_query_kp), len(per_gallery_kp)])
        query_part = per_query_des[0:min_kp_num, :]
        gallery_part = per_gallery_des[0:min_kp_num, :]
        # distance calculation in feature domain (similarity)
        dist_record.append(np.sum(
            (np.double(query_part)-np.double(gallery_part))**2)/np.prod(np.shape(query_part)))
    # find the indexes with descending similarity order
    ascend_index = sorted(range(len(dist_record)),
                          key=lambda k: dist_record[k])
    # update the results for one query
    record_all[i, :] = ascend_index
    time_e = time.time()
    print('retrieval time for query {} is {}s'.format(
        query_img_no, time_e-time_s))



# Output
f = open(r'./rank_list.txt', 'w')
for i in range(num_query):
    f.write('Q'+str(i+1)+': ')
    for j in range(len(name_gallery)):
        f.write(str(np.int32(record_all[i, j]))+' ')
    f.write('\n')
f.close()
