"""
Method 2: Convoultion Neural Network, best results, best method
"""
import numpy as np
import cv2
import glob  # for file type
import time
import os
import concurrent.futures

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity

download_path = 'C:\\Users\\lungpng2\\Documents\\datasets_4186' # Path of downloaded datasets

path_query = download_path+'\\query_4186' # Path of query images: '\\Users\\samng\\Documents\\datasets_4186\\query_4186'
name_query = glob.glob(path_query+'\\*.jpg') # File name of query images
num_query = len(name_query) # Total number of query images
path_gallery = download_path+'\\gallery_4186'
name_gallery = glob.glob(path_gallery+'\\*.jpg') # File name of gallery images
num_gallery = len(name_gallery) # Total number of gallery images

path_query_txt = download_path+'\\query_txt_4186' # Path of query images' bounding box '\\Users\\samng\\Documents\\datasets_4186\\query_txt_4186'
save_path = download_path + '\\cropped_4186' # Path of cropped image

record_all = np.zeros((num_query, len(name_gallery))) # record for similarity, initialize to 0 array
gallery_imgs_no = [x.split('\\')[-1][:-4] for x in glob.glob(path_gallery+'\\*.jpg')] # Image number of gallery images (eg. 448)
query_imgs_no = [x.split('\\')[-1][:-4] for x in glob.glob(path_query+'\\*.jpg')] # Image number of query images (eg. 2714, 776)


"""
Reference from sample code
"""
#crop the instance region. For the images containing two instances, you need to crop both of them.
def query_crop(query_img, txt_path, save_path):
    txt = np.loadtxt(txt_path)     #load the coordinates of the bounding box
    crop = query_img[int(txt[1]):int(txt[1] + txt[3]), int(txt[0]):int(txt[0] + txt[2]), :] #crop the instance region
    cv2.imwrite(save_path, crop[:,:,::-1])  #save the cropped region
    return crop

"""
ResNet101 Feature Extraction, referenced from the sample code
"""
def resnet_101_extraction(img, featsave_path):
    resnet_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    
    img_transform = resnet_transform(img)  # normalize the input image and transform it to tensor.
    img_transform = torch.unsqueeze(img_transform, 0)

    # Initialize the weights pretrained on the ImageNet dataset
    resnet101 = models.resnet101(pretrained=True)
    resnet101_feat_extractor = torch.nn.Sequential(*list(resnet101.children())[:-1])  # Define the feature extractor
    resnet101_feat_extractor.eval()  # Set the mode as evaluation
    feats = resnet101_feat_extractor(img_transform)  # Extract feature
    feats_np = feats.cpu().detach().numpy()  # Convert tensor to numpy
    np.save(featsave_path, feats_np)  # Save the feature

def feat_extractor_gallery(gallery_dir, feat_savedir):

    img_files = os.listdir(gallery_dir)

    def process_image(img_file):
        img = cv2.imread(os.path.join(gallery_dir, img_file))
        img = img[:, :, ::-1]
        # Resize image to smaller size for faster cropping
        img = cv2.resize(img, (512, 512))
        featsave_path = os.path.join(feat_savedir, img_file.split('.')[0] + '.npy')
        # Crop the image
        height, width, _ = img.shape
        crop_size = 224
        y = int((height - crop_size) / 2)
        x = int((width - crop_size) / 2)
        cropped_img = img[y:y+crop_size, x:x+crop_size, :]
        resnet_101_extraction(img, featsave_path)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_image, img_files), total=len(img_files), desc='Processing Gallery Images', dynamic_ncols=True))


def feat_extractor_query():
    # for each query image
    for i, query_img_no in enumerate (query_imgs_no):
        per_query_name = path_query + '\\' + str(query_img_no) + '.jpg'
        print('Retrieving query ' + per_query_name)
        per_query = cv2.imread(per_query_name)
        per_query = per_query[:, :, ::-1]
        txt_path = path_query_txt + '\\' + str(query_img_no) + '.txt'
        per_save_path = save_path + '\\' + str(query_img_no) + '.jpg'
        crop = query_crop(per_query, txt_path, per_save_path)
        
        featsave_path = 'C:\\Users\\lungpng2\\Documents\\datasets_4186\\query_feat\\' + query_img_no + '.npy'
        resnet_101_extraction(crop, featsave_path)
"""
Retrieval
"""
def similarity(query_feat, gallery_feat):
    query_feat = np.reshape(query_feat, (query_feat.shape[0], -1))
    gallery_feat = np.reshape(gallery_feat, (gallery_feat.shape[0], -1))
    
    sim = cosine_similarity(query_feat, gallery_feat)
    sim = np.squeeze(sim)
    return sim

def retrieval(query_feat_dir, gallery_feat_dir):
    results = {}

    for query_file in os.listdir(query_feat_dir):
        print('Retrieving for query ' + query_file)
        time_s = time.time()
        query_feat = np.load(os.path.join(query_feat_dir, query_file))
        query_name = query_file.split('.')[0]
        gallery_similarities = []
        for gallery_file in os.listdir(gallery_feat_dir):
            gallery_feat = np.load(os.path.join(gallery_feat_dir, gallery_file))
            gallery_name = gallery_file.split('.')[0]
            sim = similarity(query_feat, gallery_feat)
            gallery_similarities.append((gallery_name, sim))
        gallery_similarities.sort(key=lambda x: x[1], reverse=True)
        gallery_indices = [x[0] for x in gallery_similarities]
        results[query_name] = gallery_indices
        time_e = time.time()
        print('Processing time for query image ' + query_file +' is {}s'.format(time_e-time_s))

    return results

def main():
    # Extraction
    print('Retrieving query features...')
    feat_extractor_query()

    gallery_dir = 'C:\\Users\\lungpng2\\Documents\\datasets_4186\\gallery_4186'
    feat_savedir = 'C:\\Users\\lungpng2\\Documents\\datasets_4186\\gallery_feat'

    print('Retrieving gallery features...')
    feat_extractor_gallery(gallery_dir, feat_savedir)

    # Retrieval
    query_feat_dir = 'C:\\Users\\lungpng2\\Documents\\datasets_4186\\query_feat'
    gallery_feat_dir = 'C:\\Users\\lungpng2\\Documents\\datasets_4186\\gallery_feat'
    results = retrieval(query_feat_dir, gallery_feat_dir)

    with open('./rank_list.txt', 'w') as f:
        for query_name, gallery_indices in results.items():
            output = query_name + ': ' + ' '.join(gallery_indices)
            print(output)
            f.write(output + '\n')

    print('Results written to rank_list.txt')


if __name__=='__main__':
    main()