# -*- coding: utf-8 -*-
import sys, os

sys.path.append('../')
from pytorch.common.datasets_parsers.av_parser import AVDBParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
from data_visualization.t_sne.tsne import TSNE as torchTSNE
from scipy.spatial.distance import squareform
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances

from data_visualization.t_sne.wrapper import Wrapper
from pytorch.common.datasets_parsers.av_parser import AVDBParser



def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray


def gen_sift_features(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
    # https://ianlondon.github.io/blog/how-to-sift-opencv/
    # kp is the keypoints
    # desc is the SIFT descriptors, they're 128-dimensional vectors that we can use for our final features
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc


def gen_surf_features(gray_image):
    surf = cv2.xfeatures2d.SURF_create(400)
    kp, des = surf.detectAndCompute(gray_image, None)
    return kp, des


def gen_brief_features(image):
    star = cv2.xfeatures2d.StarDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp = star.detect(image, None)
    kp, des = brief.compute(image, kp)
    return kp, des


def desc_transform(desc, desc_name):
    desc = desc.flatten()
    print("Desc %s" % (desc_name,), desc.shape)
    return desc.tolist()



def get_data(dataset_root, file_list, max_num_clips=0):
    dataset_parser = AVDBParser(dataset_root, os.path.join(dataset_root, file_list),
                                max_num_clips=max_num_clips, ungroup=False, load_image=False)
    data = dataset_parser.get_data()
    print('clips count:', len(data))
    print('frames count:', dataset_parser.get_dataset_size())
    return data

def calc_features(data, draw=False):
    feat, targets = [], []
    for clip in data:
        if not clip.data_samples[0].labels in [7, 8]:
            continue

        # TODO: придумайте способы вычисления признаков на основе ключевых точек
        # distance between landmarks
        for i, sample in enumerate(clip.data_samples):
            if i % 8 != 0:
                continue

            sample_gray = to_gray(cv2.imread(sample.img_rel_path))
            _, desc_SIFT = gen_sift_features(sample_gray)
            desc_SIFT = desc_transform(desc_SIFT, 'SIFT')

            _, desc_SURF = gen_surf_features(sample_gray)
            desc_SURF = desc_transform(desc_SURF, 'SURF')

            _, desc_BRIEF = gen_brief_features(cv2.imread(sample.img_rel_path))
            desc_BRIEF = desc_transform(desc_BRIEF, 'BRIEF')

            lm_ref = sample.landmarks[30]  # point on the nose
            for j in range(len(sample.landmarks)):
                lm = sample.landmarks[j]
                dist.append(np.sqrt((lm_ref[0] - lm[0]) ** 2 + (lm_ref[1] - lm[1]) ** 2))
            # feat.append(dist)
            feat.append(desc_SIFT + desc_SURF + desc_BRIEF)
            targets.append(sample.labels)


            if draw:
                img = cv2.imread(sample.img_rel_path)
                for lm in sample.landmarks:
                    cv2.circle(img, (int(lm[0]), int(lm[1])), 3, (0, 0, 255), -1)
                cv2.imshow(sample.text_labels, img)
                cv2.waitKey(100)

    print('feat count:', len(feat))
    return np.asarray(feat, dtype=np.float32), np.asarray(targets, dtype=np.float32)

def draw(points2D, targets, it, save=False):
    fig = plt.figure()
    plt.scatter(points2D[:, 0], points2D[:, 1], c=targets)
    plt.axis('off')
    if save:
        plt.savefig('scatter_%s.png' % (it,), bbox_inches='tight')
        plt.close(fig)
    else:
        fig.show()
        plt.pause(5)
        plt.close(fig)

def run_tsne(feat, targets, pca_dim=50, tsne_dim=2):
    if pca_dim > 0:
        feat = PCA(n_components=pca_dim).fit_transform(feat)

    distances2 = pairwise_distances(feat, metric='euclidean', squared=True)
    # This return a n x (n-1) prob array
    pij = manifold.t_sne._joint_probabilities(distances2, 30, False)
    # Convert to n x n prob array
    pij = squareform(pij)

    i, j = np.indices(pij.shape)
    i, j = i.ravel(), j.ravel()
    pij = pij.ravel().astype('float32')
    # Remove self-indices
    idx = i != j
    i, j, pij = i[idx], j[idx], pij[idx]

    model = torchTSNE(n_points=feat.shape[0], n_dim=tsne_dim)
    w = Wrapper(model, cuda=True, batchsize=feat.shape[0], epochs=5)
    for itr in range(15):
        print("Iterations:", itr)
        w.fit(pij, i, j)
        # Visualize the results
        embed = model.logits.weight.cpu().data.numpy()
        draw(embed, targets, str(itr), save=True)
        print('*' * 100)


if __name__ == "__main__":
    # dataset dir
    base_dir = '/home/mdomrachev'
    if 1:
        train_dataset_root = base_dir + '/Data/STML/Ryerson/Video'
        train_file_list = base_dir + '/Data/STML/Ryerson/train_data_with_landmarks.txt'
    elif 0:
        train_dataset_root = base_dir + '/AFEW-VA/crop'
        train_file_list = base_dir + '/AFEW-VA/crop/train_data_with_landmarks.txt'
    elif 1:
        train_dataset_root = base_dir + '/OMGEmotionChallenge-master/omg_TrainVideos/preproc/frames'
        train_file_list = base_dir + '/OMGEmotionChallenge-master/omg_TrainVideos/preproc/train_data_with_landmarks.txt'

    # load dataset
    data = get_data(train_dataset_root, train_file_list, max_num_clips=0)

    # get features
    feat, targets = calc_features(data)

    # run t-SNE
    run_tsne(feat, targets, pca_dim=0)