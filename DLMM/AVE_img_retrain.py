# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 21:30:44 2021

@author: 98669
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from PIL import Image
import os
import random
import itertools
import numpy as np
import time
from scipy.special import softmax
import math


class Prototype(nn.Module): #原型网络，将提取的特征映射到原型空间

    def __init__(self):#初始化函数
        super(Prototype, self).__init__()#继承父类初始化函数

        self.fc1 = nn.Linear(512, 128, bias = True)
        self.fc2 = nn.Linear(128, 28, bias = False)
     
    def forward(self, x):
        out = self.fc1(x)
        # 输出的分别是经过全连接层的输出，矩阵fc1, 矩阵fc2
        fc_w1 = list(self.fc1.parameters())
        fc_w2 = list(self.fc2.parameters())
        return out,fc_w1,fc_w2

class MLP_Classifier(nn.Module):# 最终的分类器，用于输出预测概率

    def __init__(self):#初始化函数
        super(MLP_Classifier, self).__init__()#继承父类初始化函数
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 28)
    def forward(self, x):
        x = self.fc1(x)
        x = F.selu(x)
        out = self.fc2(x)
        return out 

#以下属于Resnet一个bottle的内部结构，不用看
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
#Resnet的具体结构，用来提取特征xx
class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=28):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer5 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer6 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        #self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        xx = out

        return xx

#最终定义的Resnet
def ResNet18():
    return ResNet(ResidualBlock)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 超参数设置
EPOCH = 3   #遍历数据集次数
pre_epoch = 0 # 定义已经遍历数据集的次数
BATCH_SIZE = 32      #批处理尺寸(batch_size)
r = 2  #高斯原型距离的缩放因子
trainacc_best  = 0



# 训练时的一些数据预处理
transform_train = transforms.Compose([
    #transforms.RandomCrop(128, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.Resize([128,128]),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# 标签类别
classlist = ['Church bell',
 'Male speech, man speaking',
 'Bark',
 'Fixed-wing aircraft, airplane',
 'Race car, auto racing',
 'Female speech, woman speaking',
 'Helicopter',
 'Violin, fiddle',
 'Flute',
 'Ukulele',
 'Frying (food)',
 'Truck',
 'Shofar',
 'Motorcycle',
 'Acoustic guitar',
 'Train horn',
 'Clock',
 'Banjo',
 'Goat',
 'Baby cry, infant cry',
 'Bus',
 'Chainsaw',
 'Cat',
 'Horse',
 'Toilet flush',
 'Rodents, rats, mice',
 'Accordion',
 'Mandolin']

# 模型定义-ResNet
ResNet= ResNet18().to(device)
Prototype = Prototype().to(device)
MLP_Classifier= MLP_Classifier().to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题

dir1 = '/hdd/DLMM/code/AVE/model/pretrain/net1/img/'
dir2 = '/hdd/DLMM/code/AVE/model/pretrain/net2/img/'
dir3 = '/hdd/DLMM/code/AVE/model/pretrain/net3/img/'
model1_list = os.listdir(dir1) 
model2_list = os.listdir(dir2) 
model3_list = os.listdir(dir3) 
model1_list.sort(key = lambda x: int(x[-7:-4])) 
model2_list.sort(key = lambda x: int(x[-7:-4])) 
model3_list.sort(key = lambda x: int(x[-7:-4])) 
#加载resnet，原型网络，分类网络的权重
ResNet.load_state_dict(torch.load(dir1+model1_list[-1]))
Prototype.load_state_dict(torch.load(dir2+model2_list[-1]))
MLP_Classifier.load_state_dict(torch.load(dir3+model3_list[-1]))



ResNet.train()
Prototype.train()
MLP_Classifier.train()

# 训练
if __name__ == "__main__":
    best_acc = 80  #2 初始化best test accuracy
   
    
    for epoch in range(pre_epoch, EPOCH):
        #定义学习率
        if epoch < 10:
            LR = 0.0001
        elif epoch < 30:
            LR = 0.00001
        elif epoch < 50:
            LR = 0.00001
        else :
            LR =0.00001
     
        optimizer = optim.Adam(itertools.chain(ResNet.parameters(),Prototype.parameters(),MLP_Classifier.parameters()), lr=LR,  weight_decay=2e-4)
        # print('\nEpoch: %d' % (epoch + 1))
        
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        acc = 0
        prototype_loss_all = 0.0
        
        #读数据
        dirimgfold = '/hdd/DLMM/dataset/AVE/train/img' #训练集所在的目录
        dirimgfold2 = '/hdd/DLMM/dataset/AVE/test/img' #训练集所在的目录
        
        union_all = np.load('/hdd/DLMM/code/AVE/savefile/union_retrain.npy',allow_pickle=True)  #加载保存的需要重训练的样本信息的npy
        union_all = union_all.tolist()
        union_all = union_all[0 : int(0.21*len(union_all))]
  
        retrain_label = []
        retrain_name = []
        retrain_imgloss = []
        retrain_confidence = []
        for element in union_all:
            retrain_label.append(int(element[1]))    
            output = element[-2]
            plist = softmax(output)
            loss = math.log(plist[int(element[1])])
            retrain_imgloss.append(float(loss))       #对模态课程学习排序用
            retrain_name.append(str(element[2])+'.jpg')   #图像名称 
            retrain_confidence.append(float(element[0]))
       
            
        retrain_img = sorted(zip(retrain_label, retrain_imgloss, retrain_name,retrain_confidence), key=lambda x: x[1], reverse=False)
    
        percent = min( int ( epoch / 1 + 1) * 0.1, 1 ) #每5个epoch增加20%的样本，由易到难课程学习
        
  
        retrain_img_list = retrain_img[0: int(percent*len(retrain_img))]  

        file_list=os.listdir(dirimgfold) #file_list是列表，列表元素是目录下所有训练数据的名称 
        file_list = file_list +retrain_img_list #将源域数据和待自训练的目标域数据添加到一起
        #file_list = retrain_img_list #将源域数据和待自训练的目标域数据添加到一起
        
        random.shuffle (file_list)
        trainbatch = int(len(file_list)/BATCH_SIZE)
        
        for i in range(trainbatch):
            #读取一个batch的图像和标签
            startnum = i*BATCH_SIZE
            endnum = i*BATCH_SIZE + BATCH_SIZE
            file_list_current = file_list[startnum:endnum]
            ii =  1                    
            for dir_img in file_list_current:  
                try:
                    image = Image.open(dirimgfold+'/'+dir_img)
                    
                    image = transform_train(image)
                    image = image.reshape(1,3,128,128)
                    labelname = dir_img.split('_')[0]
                    label = torch.tensor(classlist.index(labelname)).reshape(1)
                    if ii == 1:  #batchsizi=1的情况
                        batchimage = image                    
                        batchlabel =label
                        ii = ii+1
                    else:
                        batchimage = torch.cat([batchimage,image],0)
                        batchlabel = torch.cat([batchlabel,label],0)
                        ii = ii+1
                except:
                    image = Image.open(dirimgfold2+'/'+dir_img[2])
                    image = transform_train(image)
                    image = image.reshape(1,3,128,128)
                    
                    label = torch.tensor(dir_img[0]).reshape(1)
                    if ii == 1:  #batchsizi=1的情况
                        batchimage = image                    
                        batchlabel =label
                        ii = ii+1
                    else:
                        batchimage = torch.cat([batchimage,image],0)
                        batchlabel = torch.cat([batchlabel,label],0)
                        ii = ii+1
        
            batchimage, batchlabel = batchimage.to(device), batchlabel.to(device)
            optimizer.zero_grad()

            # forward + backward
            xx = ResNet(batchimage) #先经过Resnet提取特征
            outputs = MLP_Classifier(xx) #得到预测结果
            out2,fc_w1,fc_w2 = Prototype(xx) #经过原型网络的映射，得到在原型空间中的表示
            w = fc_w2[0] #原型映射矩阵的参数
            
            
            prototype_loss_batch = 0   #一个batch的原型loss
            
            for batch_num in range(BATCH_SIZE):
                for iiiii in range(28):
                    eu_distance = -1*torch.norm(out2[batch_num,::] - w[iiiii,::].reshape(128)) #负的欧式距离
                    eu_distance = eu_distance / r
                    gussian_distance = torch.exp(eu_distance)
                    if iiiii == 0:
                        max_gussian = gussian_distance
                        max_id = 0
                    if max_gussian < gussian_distance:
                        max_gussian = gussian_distance
                        max_id = iiiii
                    if batchlabel[batch_num].item() == iiiii:
                        eu_distance = -1*torch.norm(out2[batch_num,::] - w[iiiii,::].reshape(128)) #负的欧式距离
                        gussian_distance = torch.exp(eu_distance)
                        prototype_loss = -torch.log(gussian_distance.reshape(1))/28
                       
                    else:
                        prototype_loss = -torch.log(1-gussian_distance.reshape(1))/28
                    prototype_loss_batch= prototype_loss_batch + prototype_loss
                if max_id == batchlabel[batch_num]:
                    acc = acc + 1 
                    

            loss = criterion(outputs, batchlabel)    
            prototype_loss = prototype_loss_batch / BATCH_SIZE
            prototype_loss_all = prototype_loss.cpu().item() + prototype_loss_all
            weight = 0.5  # 平衡分类loss和原型loss的权重
            union_loss = weight*loss + (1-weight)*prototype_loss
                   
            union_loss.backward() #反向传播
            optimizer.step()

            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batchlabel.size(0)
            correct += predicted.eq(batchlabel.data).cpu().sum()
            # print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
            #       % (epoch + 1, (i + 1 ), sum_loss / (i + 1), 100. * correct / total))
            # print('  loss:', prototype_loss_all/(i+1),' acc:',acc/((i+1)*BATCH_SIZE))
            
            
        #保存模型
        torch.save(ResNet.state_dict(), '/hdd/DLMM/code/AVE/model/retrain/retrain_img_net1_%03d.pth' % (epoch + 1))
        torch.save(Prototype.state_dict(), '/hdd/DLMM/code/AVE/model/retrain/retrain_img_net2_%03d.pth' % (epoch + 1))
        torch.save(MLP_Classifier.state_dict(), '/hdd/DLMM/code/AVE/model/retrain/retrain_img_net3_%03d.pth' % (epoch + 1))
       
        
        trainacc = 100. * correct / total
        # if trainacc > trainacc_best:
        #     trainacc_best = trainacc
        #     trainacc_best_epoch = epoch+1
        #     print('trainacc_best:',trainacc_best)
            
            
            
            