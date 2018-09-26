import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import pandas as pd

#-------------------------------------------------------------------------------- 

class DatasetGenerator (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, CSV_PATH, transforms):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transforms
    
        #---- Open file, get image paths and labels
    

        # fileDescriptor = pd.read_csv(CSV_PATH).as_matrix()
        fileDescriptor = os.listdir(CSV_PATH)
        for file in  fileDescriptor:
            # imagePath = '../'+file[0]
            imagePath = CSV_PATH + file[0]
            # imageLabel= file[2:]
            # imageLabel = [int(i) for i in imageLabel]
            
            self.listImagePaths.append(imagePath)
            #self.listImageLabels.append(imageLabel)  

    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')
        # imageLabel= torch.FloatTensor(self.listImageLabels[index])
        imageLabel= None
        
        if self.transform != None: imageData = self.transform(imageData)
        
        return imageData, imageLabel, imagePath
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImagePaths)
    
 #-------------------------------------------------------------------------------- 
    
 if __name__=='__main__':
 	TrainPath= './oversampled_train.csv'
