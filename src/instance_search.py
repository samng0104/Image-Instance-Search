import numpy as np
import cv2
import pandas as pd
import glob  # for file type
import time

# For Bag of Words implementation
from sklearn.cluster import KMeans


def query_crop(query_img, txt_path, save_path):
    # query_img = cv2.imread(query_path)
    query_img = query_img[:,:,::-1] #bgr2rgb
    txt = np.loadtxt(txt_path)     #load the coordinates of the bounding box
    crop = query_img[int(txt[1]):int(txt[1] + txt[3]), int(txt[0]):int(txt[0] + txt[2]), :] #crop the instance region
    cv2.imwrite(save_path, crop[:,:,::-1])  #save the cropped region
    return crop

# Path of downloaded datasets
download_path = '/Users/samng/Documents/datasets_4186'

# on Windows:
download_path = 'C:/Users/Sam NG/Downloads/datasets_4186'

# Path of query images
path_query = download_path+'/query_4186'
# '/Users/samng/Documents/datasets_4186/query_4186'

# Path of query images' bounding box
path_query_txt = download_path+'/query_txt_4186'
# '/Users/samng/Documents/datasets_4186/query_txt_4186'

# Path of the gallery
path_gallery = download_path+'/gallery_4186'
# '/Users/samng/Documents/datasets_4186/gallery_4186'

# Path of cropped image
save_path = download_path + '/cropped_4186'


# File name of query images
name_query = glob.glob(path_query+'/*.jpg')
# Total number of query images
num_query = len(name_query)
# File name of gallery images   
name_gallery = glob.glob(path_gallery+'/*.jpg')
# Total number of gallery images
num_gallery = len(name_gallery)

# ORB detector as SIFT detector
sift = cv2.ORB_create()
# record for similarity, initialize to 0 array
record_all = np.zeros((num_query, len(name_gallery)))
# Image number of query images (eg. 2714, 776)
query_imgs_no = [x.split('/')[-1][:-4] for x in glob.glob(path_query+'/*.jpg')]
# Image number of gallery images (eg. 448)
gallery_imgs_no = [x.split('/')[-1][:-4]
                   for x in glob.glob(path_gallery+'/*.jpg')]

# Default Handcraft method: SIFT
# the iteration loop for query 
for i, query_img_no in enumerate(query_imgs_no):
    time_s = time.time()
    dist_record=[]
    per_query_name=path_query+'/'+str(query_img_no)+'.jpg'


    per_query=cv2.imread(per_query_name)
    # feature extraction for per query
    per_query = query_crop(per_query, path_query_txt, save_path)
    
    txt_path = path_query_txt + '/' + str(query_img_no) + '.txt'
    per_query = per_query[:,:,::-1]
    txt = np.loadtxt(txt_path)
    # quite naive, just an example
    per_query_kp, per_query_des = sift.detectAndCompute(per_query,None)

    # the iteration loop for gallery
    for j, gallery_img_no in enumerate(gallery_imgs_no):
        per_gallery_name = path_gallery+'/'+str(gallery_img_no)+'.jpg'
        per_gallery=cv2.imread(per_gallery_name)
        # feature extraction for per gallery
        per_gallery_kp, per_gallery_des = sift.detectAndCompute(per_gallery,None)
        # use part of the features to make the calculation feasible
        # quite naive, just an example
        min_kp_num=np.amin([len(per_query_kp),len(per_gallery_kp)])
        query_part=per_query_des[0:min_kp_num,:]
        gallery_part=per_gallery_des[0:min_kp_num,:]
        # distance calculation in feature domain (similarity)
        dist_record.append(np.sum((np.double(query_part)-np.double(gallery_part))**2)/np.prod(np.shape(query_part)))
    # find the indexes with descending similarity order
    ascend_index=sorted(range(len(dist_record)), key=lambda k: dist_record[k])
    # update the results for one query 
    record_all[i,:]=ascend_index
    time_e = time.time()
    print('retrieval time for query {} is {}s'.format(query_img_no, time_e-time_s))

# # Method 1: Bag of Visual Words

# # number of words
# total_words = 1000

# descriptors = []
# for path in path_gallery:
#     data = cv2.imread(name_gallery)

#     gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
#     sift = cv2.ORB_create()
#     keypoints, descriptor = sift.detectAndCompute(gray, None)
#     descriptors.append(descriptor)

# # Convert the descriptors to a single array
# descriptors = np.concatenate(descriptors)    

# # Perform k-means clustering to quantize the descriptors into visual words
# kmeans = KMeans(n_clusters=total_words, random_state=0).fit(descriptors)

# # Compute the BoW histogram for each image
# bovw = np.zeros((len(path_gallery), k))
# for i, path in enumerate(path_gallery):
#     data = cv2.imread(path)
#     gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
#     sift = cv2.ORB_create()
#     keypoints, descriptor = sift.detectAndCompute(gray, None)
#     labels = kmeans.predict(descriptor)
#     for label in labels:
#         bovw[i, label] += 1
#     bovw[i] /= np.sum(bovw[i]) # Normalize the histogram

# for path in path_query:
#     query_image = cv2.imread(path)
#     query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
#     query_sift = cv2.xfeatures2d.SIFT_create()
#     query_keypoints, query_descriptor = query_sift.detectAndCompute(query_gray, None)
#     query_labels = kmeans.predict(query_descriptor)
#     query_bovw = np.zeros((1, total_words))
#     for label in query_labels:
#         query_bovw[0, label] += 1
#     query_bovw /= np.sum(query_bovw) # Normalize the histogram

# for i in range(num_gallery):
#     similarity = np.dot(bovw[i], query_bovw.T)
#     record_all.append((similarity, name_gallery[i]))
# record_all.sort(reverse=True)
# Method 2: Convolutional Neural Networks


# Output
f = open(r'./rank_list.txt', 'w')
for i in range(num_query):
    f.write('Q'+str(i+1)+': ')
    for j in range(len(name_gallery)):
        f.write(str(np.int32(record_all[i, j]))+' ')
    f.write('\n')
f.close()
