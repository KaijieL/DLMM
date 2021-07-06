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


class Prototype(nn.Module):#自定义类 继承nn.Module

    def __init__(self):#初始化函数
        super(Prototype, self).__init__()#继承父类初始化函数

        self.fc1 = nn.Linear(512, 128, bias = True)
        self.fc2 = nn.Linear(128, 28, bias = False)

    def forward(self, x):
        out = self.fc1(x)
  
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

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=28):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer5 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer6 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        
        #self.fc = nn.Linear(512, 28)

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
        #out = self.layer6(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        xx = out
        #out = self.fc(out)

        return xx


def ResNet18():

    return ResNet(ResidualBlock)


os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
args = parser.parse_args()

# 超参数设置
EPOCH = 50   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 64      #批处理尺寸(batch_size)
trainacc_best  = 0
r = 2  #高斯原型距离的缩放因子


# 准备数据集并预处理
transform_train = transforms.Compose([

    transforms.ToTensor(),

])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    
])


# Cifar-10的标签
# Cifar-10的标签
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
ResNet = ResNet18().to(device)
Prototype = Prototype().to(device) #创建实例
MLP_Classifier = MLP_Classifier().to(device) #创建实例
# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
ResNet.train()
Prototype.train()
MLP_Classifier.train()

ResNet.load_state_dict(torch.load('C:\\Users\\98669\\Desktop\\DLMM\\retrain_model\\retrain_audio_net1.pth'))
Prototype.load_state_dict(torch.load('C:\\Users\\98669\\Desktop\\DLMM\\retrain_model\\retrain_audio_net2.pth'))
MLP_Classifier.load_state_dict(torch.load('C:\\Users\\98669\\Desktop\\DLMM\\pretrain_model\\all_audio_net3.pth'))

# 训练
if __name__ == "__main__":
    best_acc = 80  #2 初始化best test accuracy
    print("Start Training !")  # 定义遍历数据集的次数
    
    for epoch in range(pre_epoch, EPOCH):
        #定义学习率
        if epoch < 10:
            LR = 0.0001
        elif epoch < 30:
            LR = 0.0001
        elif epoch < 50:
            LR = 0.0001
        else :
            LR =0.00001
     
        optimizer = optim.Adam(itertools.chain(ResNet.parameters(),Prototype.parameters(),MLP_Classifier.parameters()), lr=LR,  weight_decay=2e-4)
        print('\nEpoch: %d' % (epoch + 1))
        
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        acc = 0
        prototype_loss_all = 0.0
        
        #读数据
        dirimgfold = 'G:\\av_newdataset\\train_img2' #训练集所在的目录
        
        union_all = np.load('C:\\Users\\98669\\Desktop\\DLMM\\savefile\\union_retrain.npy')  #加载保存的需要重训练的样本信息的npy
        union_all = union_all.tolist()
       
        retrain_label = []
        retrain_name = []
        retrain_audioloss = []
        retrain_confidence = []
        for element in union_all:
            retrain_label.append(int(element[1]))      
            retrain_audioloss.append(float(element[-2]))       #对模态课程学习排序用
            retrain_name.append(str(element[2])+'.txt')   #图像名称 
            retrain_confidence.append(float(element[0]))
            
        retrain_img = sorted(zip(retrain_label, retrain_audioloss, retrain_name,retrain_confidence), key=lambda x: x[1], reverse=False)
        
        percent = min( int ( epoch / 5 + 1) * 0.1, 1 ) #每1个epoch增加20%的样本，由易到难课程学习
   
        retrain_img_list = retrain_img[0: int(percent*len(retrain_img))]  
       
        file_list=os.listdir(dirimgfold) #file_list是列表，列表元素是目录下所有训练数据的名称 
        file_list = file_list +retrain_img_list #将源域数据和待自训练的目标域数据添加到一起
        
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
                    image = np.loadtxt('G:\\av_newdataset\\all_audio\\'+dir_img.split('.')[0]+'.txt')
                    
                    image = torch.from_numpy(image)
                    image = image.reshape(1,1,64,64)
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
                    image = np.loadtxt('G:\\av_newdataset\\all_audio\\'+dir_img[2])
                    image = torch.from_numpy(image)
                    image = image.reshape(1,1,64,64)
                    
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
            batchimage = batchimage.type(torch.cuda.FloatTensor)
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
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (i + 1 ), sum_loss / (i + 1), 100. * correct / total))
            print('  loss:', prototype_loss_all/(i+1),' acc:',acc/((i+1)*BATCH_SIZE))
            
            
        #保存模型
        torch.save(ResNet.state_dict(), 'C:\\Users\\98669\\Desktop\\DLMM\\retrain_model\\retrain_audio_net1_%03d.pth' % (epoch + 1))
        torch.save(Prototype.state_dict(), 'C:\\Users\\98669\\Desktop\\DLMM\\retrain_model\\retrain_audio_net2_%03d.pth' % (epoch + 1))
        torch.save(MLP_Classifier.state_dict(), 'C:\\Users\\98669\\Desktop\\DLMM\\retrain_model\\retrain_audio_net3_%03d.pth' % (epoch + 1))
       
        
        trainacc = 100. * correct / total
        if trainacc > trainacc_best:
            trainacc_best = trainacc
            trainacc_best_epoch = epoch+1
            print('trainacc_best:',trainacc_best)
            
            
            
            