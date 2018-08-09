# -*- coding: utf-8 -*-
import os, sys
import cv2
import random
import numpy as np
from tqdm import tqdm
import pickle

sys.path.append('../')
from pytorch.common.datasets_parsers.av_parser import AVDBParser

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from video_feature_classification.accuracy import Accuracy


import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost import DMatrix
from sklearn.metrics import f1_score
from skimage.feature import local_binary_pattern
from collections import Counter
from sklearn.metrics import classification_report


class MDMatrix(DMatrix):
    nc = 0

    def __init__(self, data, label=None, missing=None,
                 weight=None, silent=False,
                 feature_names=None, feature_types=None):
        self.nc = data.shape[1]
        super().__init__(data, label, missing, weight, silent, feature_names, feature_types)

    def num_col(self):
        return self.nc


SPACE = {
    'objective': 'multi:softmax',
    'eval_metric': 'mlogloss',
    'eta': 0.02,
    'max_depth': 4,
    'nthread': 3,
    'n_estimators': 1000,
    'num_class': 9
}

BORDER = 50
NPOINTS_LBP = 24
RADIUS_LPB = 8

import logging
import time
import uuid


def init_logging():
    fmt = logging.Formatter('%(asctime)-15s %(message)s')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    log_dir_name = "/home/mdomrachev/Data/STML/video_feature_classification/"
    log_file_name = time.strftime("%Y_%m_%d-%H_%M_%S-") + str(uuid.uuid4())[:8] + '.txt'
    logging.info('Logging to {}'.format(log_file_name))
    logfile = logging.FileHandler(os.path.join(log_dir_name, log_file_name), 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)


def get_data(dataset_root, file_list, max_num_clips=0, max_num_samples=50):
    dataset_parser = AVDBParser(dataset_root, os.path.join(dataset_root, file_list),
                                max_num_clips=max_num_clips, max_num_samples=max_num_samples,
                                ungroup=False, load_image=True)
    data = dataset_parser.get_data()
    print('clips count:', len(data))
    print('frames count:', dataset_parser.get_dataset_size())
    return data


def check_dim_desc(descs):
    mark = True
    for desc in descs:
        if desc.shape[0] != 68:
            mark = False
    return mark


def calc_features(data):
    orb = cv2.ORB_create()
    # sift = cv2.xfeatures2d.SIFT_create()
    # surf = cv2.xfeatures2d.SURF_create()
    # brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    progresser = tqdm(iterable=range(0, len(data)),
                      desc='calc video features',
                      total=len(data),
                      unit='files')

    feat, targets = [], []
    for i in progresser:
        clip = data[i]
        rm_list = []

        for i, sample in enumerate(clip.data_samples):

            dist_nose = []
            dist_mouth = []
            dist_mouth_down = []
            dist_mouth_right = []
            dist_mouth_left = []

            dist_eye_left = []
            dist_eye_right = []

            dist_brown_left = []
            dist_brown_right = []

            kp = []

            lm_nose = sample.landmarks[30]  # point on the nose

            lm_mouth = sample.landmarks[63]
            lm_mouth_down = sample.landmarks[58]
            lm_mouth_right = sample.landmarks[65]
            lm_mouth_left = sample.landmarks[61]

            lm_left_brown = sample.landmarks[20]
            lm_right_brown = sample.landmarks[25]

            lm_eye_left = sample.landmarks[42]
            lm_eye_right = sample.landmarks[44]

            img = cv2.copyMakeBorder(sample.image, BORDER, BORDER, BORDER, BORDER, cv2.BORDER_REPLICATE)

            for j in range(len(sample.landmarks)):
                lm = sample.landmarks[j]
                dist_nose.append(np.sqrt((lm_nose[0] - lm[0]) ** 2 + (lm_nose[1] - lm[1]) ** 2))

                dist_mouth.append(np.sqrt((lm_mouth[0] - lm[0]) ** 2 + (lm_mouth[1] - lm[1]) ** 2))
                dist_mouth_down.append(np.sqrt((lm_mouth_down[0] - lm[0]) ** 2 + (lm_mouth_down[1] - lm[1]) ** 2))
                dist_mouth_left.append(np.sqrt((lm_mouth_left[0] - lm[0]) ** 2 + (lm_mouth_left[1] - lm[1]) ** 2))
                dist_mouth_right.append(np.sqrt((lm_mouth_right[0] - lm[0]) ** 2 + (lm_mouth_right[1] - lm[1]) ** 2))

                dist_eye_left.append(np.sqrt((lm_eye_left[0] - lm[0]) ** 2 + (lm_eye_left[1] - lm[1]) ** 2))
                dist_eye_right.append(np.sqrt((lm_eye_right[0] - lm[0]) ** 2 + (lm_eye_right[1] - lm[1]) ** 2))


                dist_brown_left.append(np.sqrt((lm_left_brown[0] - lm[0]) ** 2 + (lm_left_brown[1] - lm[1]) ** 2))
                dist_brown_right.append(np.sqrt((lm_right_brown[0] - lm[0]) ** 2 + (lm_right_brown[1] - lm[1]) ** 2))
                p = cv2.KeyPoint(lm[0] + BORDER, lm[1] + BORDER, _size=120)
                kp.append(p)


            _, desk = orb.compute(img, kp)

            #_, desk_brief = brief.compute(img, kp)
            # _, desk_sift = sift.compute(img, kp)
            # _, desk_surf = surf.compute(img, kp)
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #lbp = local_binary_pattern(gray, NPOINTS_LBP, RADIUS_LPB, 'uniform')
            #(hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, NPOINTS_LBP + 3), range=(0, NPOINTS_LBP + 2))
            #hist = hist.astype("float") # normalize the histogram
            #hist /= (hist.sum() + 1e-7)

            if check_dim_desc([
                desk ]):
                feat.append(
                    desk.flatten().tolist() +
                    # desk_brief.flatten().tolist() +
                    # desk_sift.flatten().tolist() +
                    # desk_surf.flatten().tolist() +
                    # hist.tolist() +
                    dist_nose + 
                    dist_mouth +
                    dist_mouth_down +
                    dist_mouth_right + 
                    dist_mouth_left +
                    dist_brown_right +
                    dist_brown_left +
                    dist_eye_left + 
                    dist_eye_right
                    )
                targets.append(sample.labels)
            else:
                rm_list.append(sample)

        # feat.append(features_per_clip)
        # targets.append(clip.labels)
        # feat_len.append(len(features_per_clip))

        if rm_list:
            for sample in rm_list:
                clip.data_samples.remove(sample)

    # count_vector_frames_per_clip = Counter(feat_len)
    # optimal_size = 30
    # max_frame_vector_per_clip = 39 # max([k for k in count_vector_frames_per_clip])
    # min_frame_vector_per_clip = min([k for k in count_vector_frames_per_clip])
    # print('frame_vector_len:', len(feat[0][0]))
    # print('max frames vectors per clip:', max_frame_vector_per_clip)
    # print('min frames vectors per clip:', min_frame_vector_per_clip)
    
    #feat_transform = []
    #for f_v in feat:
    #  if len(f_v) == optimal_size:
    #      feat_transform.append([el for v in f_v for el in v])
    #  else:
    #      # f_v = f_v + [np.zeros(len(feat[0][0])) for i in range(0, max_frame_vector_per_clip - len(f_v))]
    #      feat_transform.append([el for v in f_v for el in v])
    #print('Check feat_v clips len:', Counter([len(f_v) for f_v in feat_transform]))
    
    print('objects count:', len(feat))
    print('unique labels count:', len(set(targets)))
    return np.asarray(feat, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def classification(X_train, X_test, y_train, y_test, accuracy_fn, pca_dim):
    if pca_dim > 0:
        pca = PCA(n_components=pca_dim)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        # TODO: ......... .......... ........... ......... . .............. PCA

    # shuffle
    combined = list(zip(X_train, y_train))
    random.shuffle(combined)
    X_train[:], y_train[:] = zip(*combined)

    # TODO: ........... .............. .. sklearn
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=1024, test_size=0.1)
    print('Data shape:', X_train.shape)
    y_train = y_train.reshape(-1, 1)
    y_valid = y_valid.reshape(-1, 1)

    d_train = MDMatrix(X_train, label=y_train)
    d_valid = MDMatrix(X_valid, label=y_valid)
    d_test = MDMatrix(X_test)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    model = xgb.train(SPACE,
                      d_train,
                      SPACE['n_estimators'],
                      watchlist,
                      early_stopping_rounds=50,
                      verbose_eval=10)

    y_pred = model.predict(d_test)
    accuracy_fn.by_frames(y_pred)
    accuracy_fn.by_clips(y_pred)
    # print(classification_report(y_test, y_pred, target_names=sorted(set([str(int(l)) for l in y_test]))))


if __name__ == "__main__":
    init_logging()
    experiment_name = 'exp_1'
    max_num_clips = 0 # загружайте только часть данных для отладки кода
    use_dump = False # используйте dump для быстрой загрузки рассчитанных фич из файла

    # dataset dir
    base_dir = '/home/mdomrachev'
    if 1:
        train_dataset_root = base_dir + '/Data/STML/Ryerson/Video'
        train_file_list = base_dir + '/Data/STML/Ryerson/train_data_with_landmarks.txt'
        test_dataset_root = base_dir + '/Data/STML/Ryerson/Video'
        test_file_list = base_dir + '/Data/STML/Ryerson/test_data_with_landmarks.txt'
    elif 1:
        train_dataset_root = base_dir + '/OMGEmotionChallenge-master/omg_TrainVideos/preproc/frames'
        train_file_list = base_dir + '/OMGEmotionChallenge-master/omg_TrainVideos/preproc/train_data_with_landmarks.txt'
        test_dataset_root =base_dir + '/OMGEmotionChallenge-master/omg_ValidVideos/preproc/frames'
        test_file_list = base_dir + '/OMGEmotionChallenge-master/omg_ValidVideos/preproc/valid_data_with_landmarks.txt'

    if not use_dump:
        # load dataset
        train_data = get_data(train_dataset_root, train_file_list, max_num_clips=max_num_clips)
        test_data = get_data(test_dataset_root, test_file_list, max_num_clips=max_num_clips)

        # get features
        train_feat, train_targets = calc_features(train_data)
        test_feat, test_targets = calc_features(test_data)
        

        accuracy_fn = Accuracy(test_data, experiment_name=experiment_name)

        #with open(experiment_name + '.pickle', 'wb') as f:
        #    pickle.dump([train_feat, train_targets, test_feat, test_targets, accuracy_fn], f, protocol=2)
    else:
        with open(experiment_name + '.pickle', 'rb') as f:
            train_feat, train_targets, test_feat, test_targets, accuracy_fn = pickle.load(f)

    # run classifiers
    classification(train_feat, test_feat, train_targets, test_targets, accuracy_fn=accuracy_fn, pca_dim=0)