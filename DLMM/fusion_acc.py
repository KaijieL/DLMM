# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 09:11:11 2021

@author: 98669
"""

import numpy as np
from scipy.special import softmax
import math
import matplotlib.pyplot as plt

def entropy(c):
    if max(c) == 1:
        result = 0
        return result;
    else:
        result=-1;
        if(len(c)>0):
            result=0;
        for x in c:
            result+=(-(x+0.0000000001))*math.log(x+0.0000000001,2)
        return result;



audio_compute = 1

#########################声音模态的处理

#audiolist define
audio_gth = []
audio_pro = []
audio_entroy = []
audio_prodiff = []
audio_filename = []
audio_cosine = []
audio_cosine_compare = []
audio_cosine_multi = []
audio_clc = []
audio_plist_all = []
audio_cosine_diff = []
audio_loss = []

 
d2 = np.load('/hdd/DLMM/code/AVE/savefile/audio_retrain_distance.npy',allow_pickle=True) #声音模态的评估信息



d2 = d2.tolist()
try:
    for i in range(0 , len(d2)):
        d2[i][4] = d2[i][4].split('\\')[1].split('.')[0]
except :
    pass


if audio_compute == 1:
    #所有声音的指标
    for i in range(0,len(d2)):
        #if d2[i][3] not in killfilelist2:
            audio_gth.append(d2[i][0])    #判断的对错
            output = d2[i][3][0]
            #audio_cosine_compare.append(d2[i][4][0])  #比较的余弦距离
            #audio_cosine_multi.append(d2[i][4][0]*d2[i][1][0])  #综合性的余弦距离
            audio_plist = softmax(output)
            audio_pro.append(max(audio_plist))   #最大后验概率
            audio_cosine.append(d2[i][1])  #余弦距离
            audio_entroy.append(entropy(audio_plist))   #熵
            audio_plist_sorted = sorted(audio_plist, reverse=True)
            audio_prodiff.append(audio_plist_sorted[0] - audio_plist_sorted[1])  #最大概率和第二概率的差值
            
            audio_plist=audio_plist.tolist()
            classindex = audio_plist.index(max(audio_plist))
            audio_plist_all.append(audio_plist)
            # audio_clc.append(classindex)
            audio_clc.append(d2[i][5][0])
            audio_filename.append(d2[i][4])
            audio_loss.append(d2[i][-1])
            
            # for tt in range(28):
            #     d2[i][2][tt] = abs(d2[i][2][tt]-d2[i][1])         
            # d2[i][2] = sorted(d2[i][2], reverse=False)
            # audio_cosine_diff.append(d2[i][2][1])
            
            gulist = []
            for tt in range(28):              
                gu = d2[i][2][tt]
                gulist.append(gu)
            #d2[i].append(gulist)
           
            th = gulist[d2[i][5][0]]
            #th = gulist[classindex]
            for tt in range(28):
                gulist[tt] = abs(gulist[tt]-th)         
            gulist = sorted(gulist, reverse=False) 
           
            audio_cosine_diff.append(gulist[1])


####################################################################    
#图像指标的处理   
img_gth = []
img_pro = []
img_entroy = []
img_prodiff = []
img_filename = []
img_cosine = []
img_cosine_compare = []
img_cosine_multi = []
img_clc = []
img_plist_all = []
img_cosine_diff = []
img_loss = []


d2 = np.load('/hdd/DLMM/code/AVE/savefile/img_retrain_distance.npy',allow_pickle=True) #图像模态的评估信息



d2 = d2.tolist()
try:
    for i in range(0 , len(d2)):
        d2[i][4] = d2[i][4].split('\\')[1].split('.')[0]
except:
    pass



#所有图像的指标
for i in range(0,len(d2)):
    #if d2[i][3] not in killfilelist2:
        img_gth.append(d2[i][0])    #判断的对错
        output = d2[i][3][0]
        #audio_cosine_compare.append(d2[i][4][0])  #比较的余弦距离
        #audio_cosine_multi.append(d2[i][4][0]*d2[i][1][0])  #综合性的余弦距离
        img_plist = softmax(output)
        img_pro.append(max(img_plist))   #最大后验概率
        img_cosine.append(d2[i][1])  #余弦距离
        img_entroy.append(entropy(img_plist))   #熵
        img_plist_sorted = sorted(img_plist, reverse=True)
        img_prodiff.append(img_plist_sorted[0] - img_plist_sorted[1])  #最大概率和第二概率的差值
        
        img_plist = img_plist.tolist()
        classindex = img_plist.index(max(img_plist))
        img_plist_all.append(img_plist)
        # img_clc.append(classindex)
        img_clc.append(d2[i][5][0])
        img_filename.append(d2[i][4])
        img_loss.append(d2[i][-1])
        
        gulist = []
        
        for tt in range(28):
            # gu = -1*(math.log(d2[i][2][tt]))
            # gu = np.exp(-(gu**2)/(dic3[str(tt)]))
            
            gu = -1*d2[i][2][tt]
            gu = np.exp(gu)
            #gu = d2[i][2][tt]
            gulist.append(gu)
        #d2[i].append(gulist)
        
        # d2[i][2]
        d_classindex = d2[i][2].index(max(d2[i][2]))
        if d_classindex == classindex :
            sorted(gulist, reverse=True)
            img_cosine_diff.append(gulist[0]-gulist[1])
        else:
            img_cosine_diff.append(gulist[classindex]-max(gulist))
            
            





###########################图像声音双模态融合

audio_res = sorted(zip(audio_filename,audio_gth,audio_pro,audio_cosine,audio_clc,audio_plist_all,audio_cosine_diff,audio_loss ), key=lambda x: x[0], reverse=True)
img_res = sorted(zip(img_filename ,img_gth,img_pro,img_cosine,img_clc,img_plist_all,img_cosine_diff,img_loss), key=lambda x: x[0], reverse=True)

correct = 0
for i in range(len(audio_res)):
    audio_reliability = audio_res[i][3]
    img_reliability = img_res[i][3]
    w_audio = audio_reliability / (audio_reliability +img_reliability)
    w_img = img_reliability / (audio_reliability +img_reliability)
    for j in range(len(audio_res[i][5])):
        audio_res[i][5][j] = w_audio*audio_res[i][5][j]  #根据可靠性计算权重
    for j in range(len(img_res[i][5])):
        img_res[i][5][j] = w_img*img_res[i][5][j] 
        
    pre = []  
    for j in range(len(img_res[i][5])):
        pre.append(audio_res[i][5][j] + img_res[i][5][j] )  #根据可靠性加权融合
    pre_classindex = pre.index(max(pre))
    
    if pre_classindex ==  audio_res[i][4]:
        correct = correct + 1
        
acc = correct/len(audio_res) #准确率

print('两个模态在目标域的融合概率准确率为：')
print(acc)
    
        
            
    








