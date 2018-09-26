import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
#-------------------------------------------------------------------------------- 

class DatasetGenerator (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, CSV_PATH, transforms1, transforms2, transforms3):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform1 = transforms1
        self.transform2 = transforms2
        self.transform3 = transforms3    
        #---- Open file, get image paths and labels
    

        # fileDescriptor =pd.read_csv(CSV_PATH).as_matrix()
        # if CSV_PATH.__contains__('oversampled_train'):
        #     np.random.shuffle(fileDescriptor)
        #     fileDescriptor = fileDescriptor[:300000, :]
        # for file in  fileDescriptor:
        #     imagePath = '../'+file[0]
        #     imageLabel= file[2:]
        #     imageLabel = [int(i) for i in imageLabel]
        #     self.listImagePaths.append(imagePath)
        #     self.listImageLabels.append(imageLabel)

        # fileDescriptor = pd.read_csv(CSV_PATH).as_matrix()
        fileDescriptor = os.listdir(CSV_PATH)
        for file in  fileDescriptor:
            # imagePath = '../'+file[0]
            imagePath = CSV_PATH + file
            # imageLabel= file[2:]
            # imageLabel = [int(i) for i in imageLabel]
            
            self.listImagePaths.append(imagePath)
            #self.listImageLabels.append(imageLabel)  

        # self.listImagePaths = self.listImagePaths[:15]
        # self.listImageLabels = self.listImageLabels[:15]
    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')
        # imageLabel= torch.FloatTensor(self.listImageLabels[index])
        imageLabel= 0
        
        if self.transform1 != None: imageData1 = self.transform1(imageData)
        if self.transform2 != None: imageData2 = self.transform2(imageData)
        if self.transform3 != None: imageData3 = self.transform3(imageData)      
        imageData = torch.cat((imageData1, imageData2, imageData3), 0)  
        return imageData, imageLabel,imagePath.split('/').pop().split('.')[0]
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImagePaths)
    
 #-------------------------------------------------------------------------------- 
    
if __name__ =='__main__':


    transResize= 256

    transCrop  =512
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transformList = []
    transformList.append(transforms.CenterCrop(transCrop))  
    transformList.append(transforms.Resize([transResize,transResize]))
    transformList.append(transforms.ToTensor())
    transformList.append(normalize)      
    transformSequence1=transforms.Compose(transformList)

    transCrop  =512
    transformList = []
    transformList.append(transforms.CenterCrop(transCrop))  
    transformList.append(transforms.Resize([transResize,transResize]))      
    transformList.append(transforms.ToTensor())
    transformList.append(normalize)      
    transformSequence2=transforms.Compose(transformList)

    transformList = []
    transformList.append(transforms.Resize([transResize,transResize]))        
    transformList.append(transforms.ToTensor())
    transformList.append(normalize)      
    transformSequence3=transforms.Compose(transformList)


    DG=DatasetGenerator('/media/bmi/Varghese1/bla/cataracts-2018-train/codes/micro_1_valid.csv',transformSequence1,transformSequence2,transformSequence3)
    img,classes,_= next(iter(DG))