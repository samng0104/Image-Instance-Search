"""
Method 1: Bag of Visual Word
"""
import numpy as np
import cv2
import glob  # for file type
import time


# For Bag of Words implementation
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

num_visual_words = 100

descriptor_size = 128

download_path = 'C:\\Users\\lungpng2\\Documents\\datasets_4186' # Path of downloaded datasets
path_query = download_path+'\\query_4186' # Path of query images: 'C:\temp\datasets_4186\query_4186'
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


def query_crop(query_img, txt, save_path):
    # query_img = cv2.imread(query_path)
    query_img = query_img[:, :, ::-1]  # bgr2rgb
    crop = query_img[int(txt[1]):int(txt[1] + txt[3]), int(txt[0]):int(txt[0] + txt[2]), :]  # crop the instance region
    cv2.imwrite(save_path, crop[:, :, ::-1])  # save the cropped region
    return crop

keypoints_list = []
descriptors_list = []

print('Start retrieving query images keypoints and descriptors...')
time_s = time.time()
for i, query_img_no in enumerate(query_imgs_no):
    per_query_name = path_query + '\\' + str(query_img_no) + '.jpg' # /Users/samng/Documents/datasets_4186/query_4186/2714.jpg
    per_query = cv2.imread(per_query_name) # per_query is an image
    txt_path = path_query_txt + '\\' + str(query_img_no) + '.txt' # /Users/samng/Documents/datasets_4186/query_txt_4186/2714.txt
    per_query = per_query[:, :, ::-1]
    txt = np.loadtxt(txt_path)
    # [244.  49. 262. 252.]
    per_save_path = save_path + '\\' + str(query_img_no) + '.jpg'
    # /Users/samng/Documents/datasets_4186/cropped_4186/2714.jpg
    
    # feature extraction for per query
    per_query = query_crop(per_query, txt, per_save_path)
    # Detect keypoints and compute descriptors
    per_query_kp, per_query_des = sift.detectAndCompute(per_query, None)

    keypoints_list.append(per_query_kp)
    descriptors_list.append(per_query_des)

time_e = time.time()
print('Processing time for query images is {}s'.format(time_e-time_s))

print('Start retrieving gallery images keypoints and descriptors...')
time_s = time.time()
for i, gallery_img_no in enumerate(gallery_imgs_no):
    per_gallery_name = path_gallery+'\\'+str(gallery_img_no)+'.jpg'
    per_gallery = cv2.imread(per_gallery_name)
    # feature extraction for per gallery
    per_gallery_kp, per_gallery_des = sift.detectAndCompute(per_gallery, None)
    keypoints_list.append(per_gallery_kp)
    descriptors_list.append(per_gallery_des)

time_e = time.time()
print('Processing time for gallery images is {}s'.format(time_e-time_s))


print('Flattening the descriptor list...')
descriptors_flattened = np.concatenate(descriptors_list)

print('Performing K-mean Clustering')
# Perform K-means clustering on the flattened descriptors
num_clusters = 100 # Number of clusters for visual vocabulary
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(descriptors_flattened)

# Get the visual vocabulary (cluster centers)
visual_vocabulary = kmeans.cluster_centers_

query_histogram=[]

print('Start building query images histogram...')
time_s = time.time()
for i, query_img_no in enumerate(query_imgs_no):
    # Compute the distances from descriptors to visual vocabulary
    distances = np.linalg.norm(descriptors_list[i][:, np.newaxis] - visual_vocabulary, axis=2)
    # Find the index of the nearest visual word for each descriptor
    nearest_visual_words = np.argmin(distances, axis=1)
    # Compute the histogram of visual words for the image
    histogram, _ = np.histogram(nearest_visual_words, bins=np.arange(num_clusters + 1))
    # Normalize the histogram to sum to 1
    histogram = histogram.astype(np.float32) / np.sum(histogram)
    # Append the histogram to the list of image histograms
    query_histogram.append(histogram)

time_e = time.time()
print('Processing time for query images is {}s'.format(time_e-time_s))

gallery_histograms = []

print('Start building gallery images histogram...')
time_s = time.time()

for i, gallery_img_no in enumerate(gallery_imgs_no):
    # Compute the distances from descriptors to visual vocabulary
    distances = np.linalg.norm(descriptors_list[i][:, np.newaxis] - visual_vocabulary, axis=2)
    # Find the index of the nearest visual word for each descriptor
    nearest_visual_words = np.argmin(distances, axis=1)
    # Compute the histogram of visual words for the image
    histogram, _ = np.histogram(nearest_visual_words, bins=np.arange(num_clusters + 1))
    # Normalize the histogram to sum to 1
    histogram = histogram.astype(np.float32) / np.sum(histogram)
    # Append the histogram to the list of image histograms
    gallery_histograms.append(histogram)

time_e = time.time()
print('Processing time for gallery images is {}s'.format(time_e-time_s))

query_histogram = np.array(query_histogram)
query_histogram = np.reshape(query_histogram, (1, -1))

# Compute the distance between the query image histogram and the gallery image histograms
distances = cdist(query_histogram[np.newaxis, :], gallery_histograms, metric='euclidean')


sorted_indices = np.argsort(distances)

for i, query_img_no in enumerate(query_imgs_no):
    ascend_index = sorted(range(len(distances)),key=lambda k: distances[k])
    # update the results for one query
    record_all[i, :] = ascend_index


# Output
f = open(r'./rank_list.txt', 'w')
for i in range(num_query):
    f.write('Q'+str(i+1)+': ')
    for j in range(len(name_gallery)):
        f.write(str(np.int32(record_all[i, j]))+' ')
    f.write('\n')
f.close()
