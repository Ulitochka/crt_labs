import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from skimage import io
import numpy as np
import datetime,sys
from numpy.random import randint
from STML_projects.pytorch.user_vemotions.calculateEvaluationCCC import calculateCCC
from STML_projects.pytorch.user_vemotions.net_sphere import sphere20a

from STML_projects.pytorch.user_vemotions.net import Net
from STML_projects.pytorch.user_vemotions.data_generator import OMGDataset



# Define parameters
use_cuda = torch.cuda.is_available()

lr = 0.01
bs = 2
n_epoch = 30
lr_steps = [8, 16, 24]

gd = 20 # clip gradient
eval_freq = 3
print_freq = 20
num_worker = 4
num_seg = 16
flag_biLSTM = True

classnum = 7

train_list_path = '/home/mdomrachev/Data/STML_projects/pytorch/user_vemotions/train_data_with_landmarks.csv'
val_list_path = '/home/mdomrachev/Data/STML_projects/pytorch/user_vemotions/valid_data_with_landmarks.csv'
model_path = '/home/mdomrachev/Data/STML_projects/pytorch/VEmotionNet/models/sphere20a_20171020.pth'

########################################################################################################
sphereface = sphere20a()
sphereface.load_state_dict(torch.load(model_path))
sphereface.feature =  True # remove the last fc layer because we need to use LSTM first
new_model_removed = torch.nn.Sequential(*list(sphereface.children())[:-2])
new_model_removed.add_module('fc_5', torch.nn.Linear(14, 32))

model = Net(new_model_removed)
model.cuda()

criterion = torch.nn.MSELoss()

########################################################################################################

train_loader = DataLoader(OMGDataset(train_list_path,'/home/mdomrachev/Data/STML/omg_TrainVideos/frames/'), batch_size=bs, shuffle=True, num_workers=num_worker)
val_loader = DataLoader(OMGDataset(val_list_path,'/home/mdomrachev/Data/STML/omg_ValidVideos/frames/'), batch_size=bs, shuffle=False, num_workers=num_worker)

########################################################################################################

def train(train_loader, el, criterion, optimizer, epoch):
    model.train()
    
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    
    for i, (inputs, targets, _) in enumerate(train_loader):
        
        optimizer.zero_grad()
        
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)

        inputs = torch.autograd.Variable(inputs)
        targets = torch.autograd.Variable(targets)
        
        inputs = inputs.view((-1, 3) + inputs.size()[-2:])
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        #tsn uses clipping gradient
        if gd is not None:
            total_norm = clip_grad_norm(model.parameters(),gd)
            if total_norm > gd:
                print('clippling gradient: {} with coef {}'.format(total_norm, gd/total_norm))
                
        train_loss += loss.data[0]
        
        if i % print_freq == 0:
            printoneline(dt(),'Epoch=%d Loss=%.4f\n'
                % (epoch,train_loss/(batch_idx+1)))
        batch_idx += 1



def validate(val_loader, model, criterion, epoch):
    model.eval()
    
    err_arou = 0.0
    err_vale = 0.0
    
    txt_result = open('/home/mdomrachev/Data/STML_projects/pytorch/user_vemotions/results/val_lstm_%d.csv' %epoch, 'w')
    txt_result.write('video,utterance,arousal,valence\n')

    for (inputs, targets, (vid, utter)) in val_loader:
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        inputs = torch.autograd.Variable(inputs)
        targets = torch.autograd.Variable(targets)
        
        inputs = inputs.view((-1,3)+inputs.size()[-2:])
        outputs = model(inputs)
        
        outputs = outputs.data.cpu().numpy()
        targets = targets.data.cpu().numpy()
        # print(outputs, outputs.shape)
        err_arou += np.sum((np.mean(outputs[:,0], axis=0)-targets[:,0])**2)
        err_vale += np.sum((np.mean(outputs[:,1], axis=0)-targets[:,1])**2)

        # print(err_arou, err_vale)
        
        for i in range(len(vid)):
            out = outputs
            txt_result.write('%s,%s,%f,%f\n'%(vid[i], utter[i],out[i][0],out[i][1]))
    
    txt_result.close()

    arouCCC, valeCCC = calculateCCC('/home/mdomrachev/Data/STML_projects/pytorch/user_vemotions/results/omg_ValidationVideos.csv',
                                    '/home/mdomrachev/Data/STML_projects/pytorch/user_vemotions/results/val_lstm_%d.csv'%epoch)
    return (arouCCC, valeCCC)


##############################################################################################################################################


def printoneline(*argv):
    s = ''
    for arg in argv: s += str(arg) + ' '
    s = s[:-1]
    sys.stdout.write('\r'+s)
    sys.stdout.flush()
    
def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')

def save_model(model,filename):
    state = model.state_dict()
    torch.save(state, filename)



optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) 
   
# best_arou_ccc, best_vale_ccc = validate(val_loader, model, criterion,0)

for epoch in range(n_epoch):

    print(epoch)

    if epoch in lr_steps:
        lr *= 0.1
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)    

    train(train_loader, model, criterion, optimizer, epoch)
    
    # evaluate on validation set
    if (epoch+1)%eval_freq == 0 or epoch == n_epoch-1:
        arou_ccc, vale_ccc = validate(val_loader, model, criterion, epoch)
        #print('Epoch:', epoch, "arou_ccc:", arou_ccc, "vale_ccc:", vale_ccc)
        if (arou_ccc+vale_ccc) > (best_arou_ccc + best_vale_ccc):
            best_arou_ccc = arou_ccc
            best_vale_ccc = vale_ccc
            save_model(model,'/home/mdomrachev/Data/STML_projects/pytorch/VEmotionNet/user_vemotions/models/model_lstm_{}_{}_{}.pth'.format(epoch, round(arou_ccc,4), round(vale_ccc,4)))


















