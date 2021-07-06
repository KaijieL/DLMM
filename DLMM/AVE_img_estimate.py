# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 08:58:48 2021

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
import numpy as np
import math
import matplotlib.pyplot as plt

class Prototype(nn.Module):#自定义类 继承nn.Module

    def __init__(self):#初始化函数
        super(Prototype, self).__init__()#继承父类初始化函数

        self.fc1 = nn.Linear(512, 128, bias = True)
        self.fc2 = nn.Linear(128, 28, bias = False)
        # self.model2 = nn.Sequential(
        
        #     nn.Linear(128, 28, bias = False),
        # )#自定义实例属性 model 传入自定义模型的内部构造 返回类
    def forward(self, x):
        out = self.fc1(x)
        #x传入自定义的model类 返回经过模型后的输出
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
        #out = self.fc(out)
        return xx


def ResNet18():

    return ResNet(ResidualBlock)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
args = parser.parse_args()




# 准备数据集并预处理
transform_train = transforms.Compose([
    #transforms.RandomCrop(128, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.Resize([128,128]),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.Resize([128,128]),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


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


r = 2  #缩放因子

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  

# 模型定义-ResNet
ResNet = ResNet18().to(device)
Prototype = Prototype().to(device)
MLP_Classifier = MLP_Classifier().to(device)


#加载resnet，原型网络，分类网络的权重
ResNet.load_state_dict(torch.load('C:\\Users\\98669\\Desktop\\DLMM\\pretrain_model\\img_net1.pth'))
Prototype.load_state_dict(torch.load('C:\\Users\\98669\\Desktop\\DLMM\\pretrain_model\\img_net2.pth'))
MLP_Classifier.load_state_dict(torch.load('C:\\Users\\98669\\Desktop\\DLMM\\pretrain_model\\img_net3.pth'))
ResNet.eval()
Prototype.eval()
MLP_Classifier.eval()

# 训练
if __name__ == "__main__":
    print("Waiting Test!")
    bestepoch = 0
    besttestacc = 0
    with torch.no_grad():           
        correct = 0
        total = 0

        
        dirimgfold = 'G:\\av_newdataset\\test_img2'
        file_list=os.listdir(dirimgfold)  #file_list是列表，列表元素是目录下所有目标域的样本
         
        testall_img = []
        c=0
        for dir_img in file_list:
            if c%1000 ==0:
                print(c)
            c=c+1
            currentall = []
            
            image = Image.open(dirimgfold+'/'+dir_img)
            image = transform_test(image)
            images = image.reshape(1,3,128,128)
            labelname = dir_img.split('_')[0]
            labels = torch.tensor(classlist.index(labelname)).reshape(1)
          
            
            
            images, labels = images.to(device), labels.to(device)
            feature = ResNet(images)
            outputs = MLP_Classifier(feature)
            xx128,fc_w1,fc_w2 = Prototype(feature)
           
            _, predicted = torch.max(outputs.data, 1)
            label = torch.tensor(predicted[0].cpu().numpy()).reshape(1)
            
            label = label.to(device)
            loss = criterion(outputs, label)  
            xx128=xx128.cpu().detach().numpy()[0]
            fc_w2=fc_w2[0].cpu().detach().numpy()
           
            eu_distance = -1*np.linalg.norm(xx128 - fc_w2[predicted[0].cpu().numpy(),::].reshape(128)) #负的欧式距离
            eu_distance = eu_distance / r
            gussian_distance = np.exp(eu_distance)
            
            gussian_distance_ts = [] #高斯距离的集合
            for t in range(28):                            
                eu_distance_t = np.linalg.norm(xx128 - fc_w2[t,::].reshape(128)) #负的欧式距离
                eu_distance_t = eu_distance_t / r
                gussian_distance_ts.append(eu_distance_t)


            outputs_cpu = outputs.detach().cpu().numpy()
   
            if  (predicted.cpu().numpy()) == (labels.cpu().numpy()):                    
                currentall.append(1)    #添加判断正确与否，正确为1，错误为0
            else:                   
                currentall.append(0)
            currentall.append(gussian_distance)     #添加gussian_distance
            currentall.append(gussian_distance_ts)     #添加gussian_distance的集合
            currentall.append(outputs_cpu)     #添加output_cpu，即输出概率
            currentall.append(dirimgfold+'\\'+ dir_img)     #添加filename
            currentall.append(labels.cpu().numpy())     #添加真实标签的序号，便于后续分析
            currentall.append(loss.item())     #添加loss

            testall_img.append(currentall)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        correctnum = correct.cpu().numpy().astype(float)
        acc = correctnum / total    
        print('测试分类准确率为：%f' % (acc))
        

        testall_img = np.array(testall_img)
        np.save('C:\\Users\\98669\\Desktop\\DLMM\savefile\\img_distance.npy',testall_img) #存为npy文件，后续利用
        
        