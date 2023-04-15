import numpy as np

import os
import time
from sklearn.metrics.pairwise import cosine_similarity

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

    query_feat_dir = 'C:\\Users\\lungpng2\\Documents\\datasets_4186\\query_feat'
    gallery_feat_dir = 'C:\\Users\\lungpng2\\Documents\\datasets_4186\\gallery_feat'
    print('Getting results...')
    results = retrieval(query_feat_dir, gallery_feat_dir)
    with open('rank_list.txt', 'w') as f:
        for i, gallery_indices in results.items():
            output = 'Q' + str(i+1) + ': ' + ' '.join(gallery_indices)
            f.write(output + '\n')

    print('Results written to rank_list.txt')


if __name__=='__main__':
    main()