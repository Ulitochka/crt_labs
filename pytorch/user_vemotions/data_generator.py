import pandas as pd
import numpy as np
import torch
import json
from skimage import io
from torch.utils.data import Dataset
from numpy.random import randint


num_seg = 16


class OMGDataset(Dataset):
    """OMG dataset."""

    def __init__(self, txt_file, base_path, transform=None):
        self.base_path = base_path
        self.data = pd.read_csv(txt_file, sep=",", header=None) 
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        utter = self.data.loc[idx, 0].split('/')[1]
        vid = self.data.loc[idx, 0].split('/')[0]
        img_list = self.data.loc[idx, 1]
        img_list = img_list.split('_')
        img_list = {i: f for i, f in enumerate( img_list)}
        
        num_frames = len(img_list)
        # inspired by TSN's pytorch code
        average_duration = num_frames // num_seg
        if num_frames>num_seg:
            offsets = np.multiply(list(range(num_seg)), average_duration) + randint(average_duration, size=num_seg)
        else:
            tick = num_frames / float(num_seg)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_seg)])
        # print('offsets:', offsets)

        final_list = [img_list[i] for i in offsets]
        
        # stack images within a video in the depth dimension
        for i,ind in enumerate(final_list):
            # 0e02ee3c5_3\utterance_19.mp4\286.jpg
            image = io.imread(self.base_path+'%s/%s/%s' % (vid,utter,ind,)).astype(np.float32)
            image = torch.from_numpy(((image - 127.5)/128).transpose(2,0,1))
            if i==0:
                images = image
            else:
                images = torch.cat((images,image), 0)
        
        label = torch.from_numpy(np.array([self.data.iloc[idx,2], self.data.iloc[idx,3]]).astype(np.float32))

        if self.transform:
            image = self.transform(image)
        
        return (images, label, (vid, utter,))