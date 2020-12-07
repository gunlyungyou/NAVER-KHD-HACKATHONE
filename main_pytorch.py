import os, sys
import argparse
import time
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch_optimizer as optim
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import torch.nn.functional as F
#import torchvision
#import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import nsml
from nsml.constants import DATASET_PATH, GPU_NUM
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau

IMSIZE = 120, 60
VAL_RATIO = 0.2
RANDOM_SEED = 1234

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(dir_name, 'model'))
        print('model saved!')
    def load(dir_name):
        model.load_state_dict(torch.load(os.path.join(dir_name, 'model')))
        model.eval()
        print('model loaded!')
    def infer(data):  ## test mode
        X = ImagePreprocessing(data)
        X = np.array(X)
        X = np.expand_dims(X, axis=1)
        ##### DO NOT CHANGE ORDER OF TEST DATA #####
        with torch.no_grad():
            X = torch.from_numpy(X).float().to(device)
            pred = model.forward(X)
            prob, pred_cls = torch.max(pred, 1)
            pred_cls = pred_cls.tolist()
            #pred_cls = pred_cls.data.cpu().numpy()
        print('Prediction done!\n Saving the result...')
        return pred_cls
    nsml.bind(save=save, load=load, infer=infer)

def DataLoad(imdir):
    impath = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(imdir) for f in files if all(s in f for s in ['.jpg'])]
    img = []
    lb = []
    print('Loading', len(impath), 'images ...')
    for i, p in enumerate(impath):
        img_whole = cv2.imread(p, 0)
        h, w = img_whole.shape
        h_, w_ = h, w//2
        l_img = img_whole[:, w_:2*w_]
        r_img = img_whole[:, :w_]
        _, l_cls, r_cls = os.path.basename(p).split('.')[0].split('_')
        if l_cls=='0' or l_cls=='1' or l_cls=='2' or l_cls=='3':
            img.append(l_img);      lb.append(int(l_cls))
        if r_cls=='0' or r_cls=='1' or r_cls=='2' or r_cls=='3':
            img.append(r_img);      lb.append(int(r_cls))
        
        # augmented image
        l_img = img_whole[:, :w_]
        r_img = img_whole[:, w_:2*w_]
        _, r_cls, l_cls = os.path.basename(p).split('.')[0].split('_')
        if l_cls=='1' or l_cls=='2' or l_cls=='3':
            img.append(cv2.flip(l_img, 1));      lb.append(int(l_cls))
            img.append(cv2.flip(l_img, 0));      lb.append(int(l_cls))
        if r_cls=='1' or r_cls=='2' or r_cls=='3':
            img.append(cv2.flip(r_img, 1));      lb.append(int(r_cls))
            img.append(cv2.flip(r_img, 0));      lb.append(int(r_cls))

    print(len(img), 'data with label 0-3 loaded!')
    return img, lb


def ImagePreprocessing(img):
    h, w = 120, 60
    print('Preprocessing ...')
    for i, im, in enumerate(img):
        tmp = cv2.resize(im, dsize=(w, h), interpolation=cv2.INTER_AREA)

        #image contrast stretching
        # 특정밝기영영에 영상픽셀의 밝기값이 집중되어 있는 것을 퍼트려 가시도를 좋게하는 방법
        # '''
        # alpha = 1.0 # simple contrast control 1.0 ~ 3.0
        # beta = 0 # simple brightness control 0 ~ 100
        # '''
        #alpha = 1.2
        #beta = 25
        #tmp = tmp*alpha + beta
        #histogram equalization
        hist, bins = np.histogram(tmp.flatten(), 256,[0,256])
        cdf = hist.cumsum()
        # cdf의 값이 0인 경우는 mask처리를 하여 계산에서 제외
        # mask처리가 되면 Numpy 계산에서 제외가 됨
        #아래는 cdf array에서 값이 0인 부분을 mask처리함
        cdf_m = np.ma.masked_equal(cdf,0)
        #History Equalization 공식
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        # Mask처리를 했던 부분을 다시 0으로 변환
        cdf = np.ma.filled(cdf_m,0).astype('uint8')
        tmp = cdf[tmp]
        #clahe stretching
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #tmp = clahe.apply(tmp)
        tmp = tmp/ 255.
        #global norm
        mean = 0.3499
        std = 0.2557
        tmp = (tmp - mean)/std
        img[i] = tmp[20:100, :]
        # print(f"img[{i}].size():: {img[i]}")
        

    print(len(img), 'images processed!')
    return img

def ParserArguments(args):
    # Setting Hyperparameters
    args.add_argument('--epoch', type=int, default=30)          # epoch 수 설정
    args.add_argument('--batch_size', type=int, default=128)      # batch size 설정
    args.add_argument('--learning_rate', type=float, default=0.01)  # learning rate 설정
    args.add_argument('--num_classes', type=int, default=4)     # 분류될 클래스 수는 4개
    # DO NOT CHANGE (for nsml)
    args.add_argument('--mode', type=str, default='train', help='submit일 때 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()
    return config.epoch, config.batch_size, config.num_classes, config.learning_rate, config.pause, config.mode


class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.cbam = CBAM(512, 16)
        self.linear = nn.Linear(3072, 4)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(2860, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 2860)

        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, images):
        # images = torch.squeeze(images)
#        images = self.stn(images)

        images = images.repeat(1,3,1,1)

        features = self.resnet(images)
#        features = self.cbam(features)

        features = features.reshape(features.size(0), -1)
        output = self.linear(features)

        return output

'''
class PNSDataset(Dataset):
    def __init(self, x, y):
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len
'''
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    print(GPU_NUM)
    nb_epoch, batch_size, num_classes, learning_rate, ifpause, ifmode = ParserArguments(args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #####   Model   #####
    model = Resnet18()
    #model.double()
    model.to(device)
    #####    opimizer   ####
    optimizer = optim.AdaMod(model.parameters(), lr=learning_rate)
    # optimizer = optim.RAdam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    nSamples = [2500, 720, 300, 180]
    normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    normedWeights = torch.FloatTensor(normedWeights).to(device)
    criterion = nn.CrossEntropyLoss(weight=normedWeights)
    # criterion = nn.CrossEntropyLoss()
    
    bind_model(model)
    
    
    
    if ifpause:  ## for test mode
        print('Inferring Start ...')
        nsml.paused(scope=locals())
    if ifmode == 'train':  ## for train mode
        print('Training start ...')
        # 자유롭게 작성
        images, labels = DataLoad(imdir=os.path.join(DATASET_PATH, 'train'))
        images = ImagePreprocessing(images)
        images = np.array(images)
        images = np.expand_dims(images, axis=1)
        labels = np.array(labels)
        dataset = TensorDataset(torch.from_numpy(images).float(), torch.from_numpy(labels).long())
        subset_size = [len(images) - int(len(images) * VAL_RATIO),int(len(images) * VAL_RATIO)]
        tr_set, val_set = random_split(dataset, subset_size)
        batch_train = DataLoader(tr_set, batch_size=batch_size, shuffle=True)
        batch_val = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        print(f"batch_val::{batch_val} type::{type(batch_val)} ")
        #####   scheduler    ####
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2, verbose=True)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma= 0.99) 
        print(f"scheduler:: {scheduler}")
        #####   Training loop   #####
        STEP_SIZE_TRAIN = len(images) // batch_size
        print('\n\n STEP_SIZE_TRAIN= {}\n\n'.format(STEP_SIZE_TRAIN))
        t0 = time.time()
        for epoch in range(nb_epoch):
            t1 = time.time()
            print('Model fitting ...')
            print('epoch = {} / {}'.format(epoch + 1, nb_epoch))
            print('check point = {}'.format(epoch))
            a, a_val, tp, tp_val = 0, 0, 0, 0
            for i, (x_tr, y_tr) in enumerate(batch_train):
                x_tr, y_tr = x_tr.to(device), y_tr.to(device)
                optimizer.zero_grad()
                pred = model(x_tr)
                loss = criterion(pred, y_tr)
                # loss = focal_loss(pred, y_tr)
                loss.backward()
                optimizer.step()
                prob, pred_cls = torch.max(pred, 1)
                a += y_tr.size(0)
                tp += (pred_cls == y_tr).sum().item()
                print("Batch: {}".format(i))
            scheduler.step(loss)
            y_pred = []
            y_true = []
            with torch.no_grad():
                for j, (x_val, y_val) in enumerate(batch_val):
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    # print(f"y_val::{y_val}")
                    y_true.extend(y_val.tolist())
                    pred_val = model(x_val)
                    loss_val = criterion(pred_val, y_val)
                    prob_val, pred_cls_val = torch.max(pred_val, 1)
                    y_pred.extend(pred_cls_val.tolist())
                    # print(f"prob_val:: {prob_val} pred_cls_val:: {pred_cls_val}")
                    print(f"len:: {len(y_pred)} pred_y::{y_pred}")
                    a_val += y_val.size(0)
                    tp_val += (pred_cls_val == y_val).sum().item()
            acc = tp / a
            acc_val = tp_val / a_val
            print(f"Confusion Matris:\n{confusion_matrix(y_true, y_pred)}")
            cf_report = classification_report(y_true, y_pred, output_dict=True)
            print("type:: {} Classificcation Report: \n{}".format(type(cf_report),cf_report))
            print("Learning Rate: {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
            
            lb_score = 0
            for label, data in cf_report.items():
                if label in ['0','1','2','3']:
                    print(f"label::{label} f1-score::{data['f1-score']}")
                    lb_score += (int(label)+1) * data["f1-score"]
                else:
                    print(f"label is not decimal::{label}")
                    break   
            lb_score /= 10
            print("  * loss = {}\n  * acc = {}\n  * loss_val = {}\n  * acc_val = {}\n  * lb_score = {}".format(loss.item(), acc, loss_val.item(), acc_val, lb_score))
            nsml.report(summary=True, step=epoch, epoch_total=nb_epoch, loss=loss.item(), acc=acc, val_loss=loss_val.item(), val_acc=acc_val, lb_score=lb_score)
            nsml.save(epoch)
            print('Training time for one epoch : %.1f\n' % (time.time() - t1))
        print('Total training time : %.1f' % (time.time() - t0))
