from PIL import Image
from torchvision import transforms 


a= Image.open('/media/bmi/Varghese1/bla/cataracts-2018-train/codes/query.png').convert('RGB')
transResize= 256

transCrop=512
# normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = []

transformList.append(transforms.CenterCrop(transCrop))
transformList.append(transforms.Resize([transResize,transResize]))        
# transformList.append(transforms.RandomHorizontalFlip())
# transformList.append(transforms.ToTensor())
# transformList.append(normalize)      
transformSequence=transforms.Compose(transformList)


transCrop= 1024
transformList = []
# transformList.append(transforms.Resize([transResize,transResize]))
transformList.append(transforms.CenterCrop(transCrop))  
transformList.append(transforms.Resize([transResize,transResize]))      
# transformList.append(transforms.RandomHorizontalFlip())
# transformList.append(transforms.ToTensor())
# transformList.append(normalize)      
transformSequence2=transforms.Compose(transformList)

# transform2= transforms.Resize([512,512])

# transCrop= 1024
transformList = []
# transformList.append(transforms.Resize([transResize,transResize]))
# transformList.append(transforms.CenterCrop(transCrop))  
transformList.append(transforms.Resize([transResize,transResize]))      
# transformList.append(transforms.RandomHorizontalFlip())
# transformList.append(transforms.ToTensor())
# transformList.append(normalize)      
transformSequence3=transforms.Compose(transformList)





t1= transformSequence(a)
t2= transformSequence2(a)
t3=transformSequence3(a)

# import os
# import numpy as np

# import torch
# import torch.nn as nn
# import torch.backends.cudnn as cudnn

# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from sklearn.metrics import roc_auc_score
# from torchvision.models.densenet import model_urls
# from torchvision.models.resnet import model_urls as resnetmodel_urls

# import torchvision

# class DenseNet121(nn.Module):

#     def __init__(self, classCount, isTrained,num_channel=12):
    
#         super(DenseNet121, self).__init__()
#         model_urls['densenet121'] = model_urls['densenet121'].replace('https://', 'http://')
#         self.first_conv  =nn.Sequential(nn.BatchNorm2d(num_channel),nn.Conv2d(num_channel, 3, kernel_size=3, padding=1))
#         self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)
#         self.features    = self.densenet121.features
#         kernelCount = self.densenet121.classifier.in_features
#         self.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

#     def forward(self, x):
#         x= self.first_conv(x)
#         x= self.features(x)
#         x= nn.functional.adaptive_avg_pool2d(x,(1,1)).view(x.size(0),-1)
#         x= self.classifier(x)
#         return x



# from torch.autograd import Variable
# net=DenseNet121(2,True).cuda()
# a= Variable(torch.rand(4,12,512,512)).cuda()
# b=net(a)

# t2= transform2(a)

