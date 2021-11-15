# -*- coding: utf-8 -*-


import os



            
################################################
print('开始图像模态预训练')
p = os.system('python AVE_img_pretrain.py')
if (p != 0):
    print('wrong')
else:
    print('true,finish')



################################################
print('开始声音模态预训练')
p = os.system('python AVE_audio_pretrain.py')
if (p != 0):
    print('wrong')
else:
    print('true,finish')



################################################
print('开始图像模态跨域评估')
p = os.system('python AVE_img_estimate.py')
if (p != 0):
    print('wrong')
else:
    print('true,finish')


################################################
print('开始声音模态跨域评估')
p = os.system('python AVE_audio_estimate.py')
if (p != 0):
    print('wrong')
else:
    print('true,finish')


################################################
print('生成增量学习样例')
p = os.system('python Pseudo_label_generation.py')
if (p != 0):
    print('wrong')
else:
    print('true,finish')


################################################
print('开始图像模态异步学习')
p = os.system('python AVE_img_retrain.py')
if (p != 0):
    print('wrong')
else:
    print('true,finish')



################################################
print('开始声音模态异步学习')
p = os.system('python AVE_audio_retrain.py')
if (p != 0):
    print('wrong')
else:
    print('true,finish')

################################################
print('开始评估异步学习后图像模态的效果')
p = os.system('python AVE_img_test.py')
if (p != 0):
    print('wrong')
else:
    print('true,finish')



################################################
print('开始评估异步学习后声音模态的效果')
p = os.system('python AVE_audio_test.py')
if (p != 0):
    print('wrong')
else:
    print('true,finish')



################################################
print('开始评估异步学习后融合两个模态的效果')
p = os.system('python fusion_acc.py')
if (p != 0):
    print('wrong')
else:
    print('true,finish')
