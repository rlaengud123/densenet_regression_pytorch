#nvh
#%%
import argparse
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset

import os
import sys
import math

import shutil

import setproctitle
import pandas as pd
from skimage import io, transform, color
import matplotlib.pyplot as plt
import cv2
import numpy as np

from models import densenet
import make_graph
import easydict
args = easydict.EasyDict({
    "batchSz": 16,
    "nEpochs": 300,
    "cuda": True,
    "save": 'work/densenet.base',
    "seed": 1,
    "opt": 'adam'
    })
#%%
df = pd.read_csv('./Inteiror_results_all.csv')
df.drop([0,1,37],axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)
for i in range(len(df)):
    df['Name[tree]'][i] = 'run{}.png'.format(i+1)
#%%
class CustomDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, df, root_dir, transform=None):
        """
        Args:
            csv_file (string): csv 파일의 경로
            root_dir (string): 모든 이미지가 존재하는 디렉토리 경로
            transform (callable, optional): 샘플에 적용될 Optional transform
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, 'APLR_iso_view',self.df.iloc[idx, 0])
        image1 = cv2.imread(img_name)

        img_name = os.path.join(self.root_dir, 'front_view',self.df.iloc[idx, 0])
        image2 = cv2.imread(img_name)

        img_name = os.path.join(self.root_dir, 'mirror_front_view',self.df.iloc[idx, 0])
        image3 = cv2.imread(img_name)

        img_name = os.path.join(self.root_dir, 'side_view',self.df.iloc[idx, 0])
        image4 = cv2.imread(img_name)

        img_name = os.path.join(self.root_dir, 'top_view',self.df.iloc[idx, 0])
        image5 = cv2.imread(img_name)

        y = float(self.df.iloc[idx,11])
        sample = {'image1': image1, 'image2' : image2, 'image3' : image3, 'image4' : image4, 'image5' : image5, 'y': y}

        if self.transform:
            sample['image1'] = self.transform(sample['image1'])
            sample['image2'] = self.transform(sample['image2'])
            sample['image3'] = self.transform(sample['image3'])
            sample['image4'] = self.transform(sample['image4'])
            sample['image5'] = self.transform(sample['image5'])

        return sample
#%%
def main(df):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--batchSz', type=int, default=64)
    # parser.add_argument('--nEpochs', type=int, default=300)
    # parser.add_argument('--cuda', default=True)
    # parser.add_argument('--save')
    # parser.add_argument('--seed', type=int, default=1)
    # parser.add_argument('--opt', type=str, default='sgd',
    #                     choices=('sgd', 'adam', 'rmsprop'))
    # args = parser.parse_args()

    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    # args.save = args.save or 'work/densenet.base'
    setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    normTransform = transforms.Normalize(normMean, normStd)

    trainTransform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normTransform
    ])

    testTransform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        normTransform
    ])

    df = df.sample(frac=1).reset_index(drop=True)

    # df = df[:int(len(df)*0.1)]

    train_df = df[:int(len(df)*0.7)]
    test_df = df[int(len(df)*0.7):].reset_index(drop=True)

    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    trainset = CustomDataset(df=train_df,root_dir=r'F:\FDAI\NVH\NVH_data\AI_Exterior_Windnoise_image',transform=trainTransform)

    testset = CustomDataset(df=test_df,root_dir=r'F:\FDAI\NVH\NVH_data\AI_Exterior_Windnoise_image',transform=testTransform)

    trainLoader = DataLoader(trainset, batch_size=args.batchSz, shuffle=True, **kwargs)

    testLoader = DataLoader(testset, batch_size=args.batchSz, shuffle=False, **kwargs)

    net = densenet.densenet121()
    net = nn.DataParallel(net)

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    if args.cuda:
        net = net.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1,
                            momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')

    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF)
        test(args, epoch, net, testLoader, optimizer, testF)
        torch.save(net, os.path.join(args.save, 'latest.pth'))
        os.system('./plot.py {} &'.format(args.save))

    trainF.close()
    testF.close()
#%%
def train(args, epoch, net, trainLoader, optimizer, trainF):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    criterion = nn.MSELoss()
    
    for batch_idx, data in enumerate(trainLoader, 0):
        if args.cuda:
            inputs1, inputs2, inputs3, inputs4, inputs5, y = data['image1'].cuda(), data['image2'].cuda(), data['image3'].cuda(), data['image4'].cuda(), data['image5'].cuda(), data['y'].cuda()
        inputs1, inputs2, inputs3, inputs4, inputs5, y = Variable(inputs1), Variable(inputs2), Variable(inputs3), Variable(inputs4), Variable(inputs5), Variable(y)
        optimizer.zero_grad()
        output = net(inputs1)
        output = torch.reshape(output, (-1,))
        loss = criterion(output.float(), y.float())
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        output1 = net(inputs1)
        output1 = torch.reshape(output1, (-1,))
        loss1 = criterion(output1.float(), y.float())
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)

        optimizer.zero_grad()
        output2 = net(inputs2)
        output2 = torch.reshape(output2, (-1,))
        loss2 = criterion(output2.float(), y.float())
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)

        optimizer.zero_grad()
        output3 = net(inputs3)
        output3 = torch.reshape(output3, (-1,))
        loss3 = criterion(output3.float(), y.float())
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)


        optimizer.zero_grad()
        output4 = net(inputs4)
        output4 = torch.reshape(output4, (-1,))
        loss4 = criterion(output4.float(), y.float())
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)

        optimizer.zero_grad()
        output5 = net(inputs5)
        output5 = torch.reshape(output5, (-1,))
        loss5 = criterion(output5.float(), y.float())
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)


        loss = (loss1 + loss2 + loss3 + loss4 + loss5)/5

        loss.backward()
        optimizer.step()

        nProcessed += len(data['image1'])
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tMSE: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader), loss))

        trainF.write('{},{}\n'.format(partialEpoch, loss))
        trainF.flush()
        
def MAPE(y_true, y_pred): 
       y_true, y_pred = y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
       return np.abs((y_true - y_pred) / y_true)


def test(args, epoch, net, testLoader, optimizer, testF):
    net.eval()
    test_loss = 0
    mape = 0
    
    for batch_idx, data in enumerate(testLoader, 0):
        if args.cuda:
            inputs1, inputs2, inputs3, inputs4, inputs5, y = data['image1'].cuda(), data['image2'].cuda(), data['image3'].cuda(), data['image4'].cuda(), data['image5'].cuda(), data['y'].cuda()
        inputs1, inputs2, inputs3, inputs4, inputs5, y = Variable(inputs1), Variable(inputs2), Variable(inputs3), Variable(inputs4), Variable(inputs5), Variable(y)

        output1 = net(inputs1)
        output2 = net(inputs2)
        output3 = net(inputs3)
        output4 = net(inputs4)
        output5 = net(inputs5)

        output =  np.sum(output1, output2, output3, output4, output5)/5

        test_loss = np.sum(MAPE(output.float(), y.float()))
        mape += test_loss

    mape = mape/len(testLoader.dataset)*100
        
    print('\nTest MAPE: {}%\n'.format(mape))

    testF.write('{},{}\n'.format(epoch, mape))
    testF.flush()

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150: lr = 1e-1
        elif epoch == 150: lr = 1e-2
        elif epoch == 225: lr = 1e-3
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
#%%
if __name__=='__main__':
    main(df)


# %%
