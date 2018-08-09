# -*- coding: utf-8 -*-
import sys
import random

random.seed(1024)
import numpy as np
from tqdm import tqdm
import pickle

sys.path.append('../')
from STML.pytorch.common.datasets_parsers.av_parser import AVDBParser

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA

from STML.pytorch.common.datasets_parsers.av_parser import AVDBParser
from STML.audio_feature_classification.voice_feature_extraction import OpenSMILE
from STML.audio_feature_classification.accuracy import Accuracy, Accuracy_regression

import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost import DMatrix

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer


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
    'nthread': 5,
    'n_estimators': 1000,
    'num_class': 9
}

grid_1 = {"n_estimators": [1000, 1500],
          "criterion": ["gini", "entropy"],
          "max_features": ['sqrt', 'log2', 0.2, 0.5, 0.8],
          "max_depth": [3, 4, 6],
          "min_samples_split": [2, 5, 20, 50]}


def get_data(dataset_root, file_list, max_num_clips=0):
    dataset_parser = AVDBParser(dataset_root, file_list,
                                max_num_clips=max_num_clips)
    data = dataset_parser.get_data()
    print('clips count:', len(data))
    print('frames count:', dataset_parser.get_dataset_size())
    return data


def calc_features(data, opensmile_root_dir, opensmile_config_path):
    vfe = OpenSMILE(opensmile_root_dir, opensmile_config_path)

    progresser = tqdm(iterable=range(0, len(data)),
                      desc='calc audio features',
                      total=len(data),
                      unit='files')

    feat, targets = [], []
    for i in progresser:
        clip = data[i]

        try:
            voice_feat = vfe.process(clip.wav_rel_path)
        except:
            print('error calc voice features!')
            data.remove(clip)

        feat.append(voice_feat)
        targets.append(clip.labels)

    print('feat count:', len(feat))
    return np.asarray(feat, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def f_(y_true, y_pred): return f1_score(y_true, y_pred, average='weighted')


def classification(X_train, X_test, y_train, y_test, accuracy_fn, pca_dim=100):
    if pca_dim > 0:
        pca_model = PCA(n_components=min(pca_dim, X_train.shape[1])).fit(X_train)
        X_train = pca_model.transform(X_train)
        X_test = pca_model.transform(X_test)

    # shuffle
    combined = list(zip(X_train, y_train))
    random.shuffle(combined)
    X_train[:], y_train[:] = zip(*combined)

    # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=1024, test_size=0.1)
    print('Data shape:', X_train.shape)
    # y_train = y_train.reshape(-1, 1)
    # y_valid = y_valid.reshape(-1, 1)

    # d_train = MDMatrix(X_train, label=y_train)
    # d_valid = MDMatrix(X_valid, label=y_valid)
    # d_test = MDMatrix(X_test)

    # watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    # model = xgb.train(SPACE,
    #                 d_train,
    #                  SPACE['n_estimators'],
    #                  watchlist,
    #                  early_stopping_rounds=50,
    #                  verbose_eval=10)

    RF = RandomForestClassifier()

    score = make_scorer(f_)
    model = GridSearchCV(RF, grid_1, n_jobs=5, cv=2, verbose=1, scoring=score)
    model.fit(X_train, y_train)

    print('best_score:', model.best_score_)
    print('best_params:', model.best_params_)

    y_pred = model.predict(X_test)
    accuracy_fn.by_clips(y_pred)


if __name__ == "__main__":
    experiment_name = 'exp_1'
    max_num_clips = 100  # .......... ...... ..... ...... ... ....... ....
    use_dump = True  # ........... dump ... ....... ........ ............ ... .. .....

    # dataset dir
    base_dir = '/home/mdomrachev/Data/STML'
    if 1:
        train_dataset_root = base_dir + '/Ryerson/Video'
        train_file_list = base_dir + '/Ryerson/train_data_with_landmarks.txt'
        test_dataset_root = base_dir + '/Ryerson/Video'
        test_file_list = base_dir + '/Ryerson/test_data_with_landmarks.txt'
    elif 1:
        train_dataset_root = base_dir + '/OMGEmotionChallenge-master/omg_TrainVideos/preproc/frames'
        train_file_list = base_dir + '/OMGEmotionChallenge-master/omg_TrainVideos/preproc/train_data_with_landmarks.txt'
        test_dataset_root = base_dir + '/OMGEmotionChallenge-master/omg_ValidVideos/preproc/frames'
        test_file_list = base_dir + '/OMGEmotionChallenge-master/omg_ValidVideos/preproc/valid_data_with_landmarks.txt'

    # opensmile configuration
    opensmile_root_dir = '/home/mdomrachev/Data/STML/opensmile/opensmile-2.3.0/inst'
    # TODO: ................... . .......... ................. ....... .......... OpenSmile
    opensmile_config_path = '/home/mdomrachev/Data/STML/opensmile/opensmile-2.3.0/config/ComParE_2016.conf'

    if not use_dump:
        # load dataset
        train_data = get_data(train_dataset_root, train_file_list, max_num_clips=max_num_clips)
        test_data = get_data(test_dataset_root, test_file_list, max_num_clips=max_num_clips)

        # get features
        train_feat, train_targets = calc_features(train_data, opensmile_root_dir, opensmile_config_path)
        test_feat, test_targets = calc_features(test_data, opensmile_root_dir, opensmile_config_path)

        accuracy_fn = Accuracy(test_data, experiment_name=experiment_name)

        with open(experiment_name + '.pickle', 'wb') as f:
            pickle.dump([train_feat, train_targets, test_feat, test_targets, accuracy_fn], f, protocol=2)
    else:
        with open(experiment_name + '.pickle', 'rb') as f:
            train_feat, train_targets, test_feat, test_targets, accuracy_fn = pickle.load(f)

    # run classifiers
    classification(train_feat, test_feat, train_targets, test_targets, accuracy_fn=accuracy_fn, pca_dim=0)
