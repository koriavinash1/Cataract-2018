import os
import numpy as np
import time
import sys


from Glimpse_Like_Networks import DenseNet121
from Glimpse_Like_Networks import DenseNet169
from Glimpse_Like_Networks import DenseNet161
from Glimpse_Like_Networks import DenseNet201

from Trainer import Trainer


import json
import pandas as pd

# from Inference import Inference

Trainer = Trainer()


nclasses = 21

#--------------------------------------------------------------------------------

def main (nnClassCount=nclasses):
    # "Define Architectures and run one by one"

    nnArchitectureList = [
                            {
                                'name': 'DenseNet201',
                                'model' : DenseNet201(classCount = nclasses, isTrained = True),
                                'TrainPath': './oversampled_train.csv',
                                'ValidPath': './combined_valid.csv',
                                'ckpt': '../models/model-m-17062018-105912-DenseNet201_loss = 0.008101757148136384_acc = 0.970111232357_best_loss.pth.tar'
                            },

                            {
                                'name': 'DenesNet169',
                                'model' : DenseNet169(classCount = nclasses, isTrained = True),
                                'TrainPath': './oversampled_train.csv',
                                'ValidPath': './combined_valid.csv',
                                'ckpt': None
                            }
                        ]

    for nnArchitecture in nnArchitectureList:
        runTrain(nnArchitecture=nnArchitecture)




#--------------------------------------------------------------------------------

def runTrain(nnArchitecture = None):

    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    TrainPath = nnArchitecture['TrainPath']
    ValidPath = nnArchitecture['ValidPath']
    nnClassCount = nclasses

    #---- Training settings: batch size, maximum number of epochs
    trBatchSize = 4
    trMaxEpoch = 25
    transResize=256
    transCrop  =224
    learningRate = 0.0001
    print ('Training NN architecture = ', nnArchitecture['name'])

    info_dict = {
                'batch_size': trBatchSize,
                'architecture':nnArchitecture['name'] ,
                'number of epochs':trMaxEpoch,
                'learningRate':learningRate,
                'train path':TrainPath, 
                'valid_path':ValidPath,
                'number of classes':nclasses,
                'Date-Time':    timestampLaunch
    } 

    with open('../models/config'+str(timestampLaunch)+'.txt','w') as outFile:
        json.dump(info_dict, outFile)
    Trainer.train(TrainPath, ValidPath, nnArchitecture, nnClassCount,transResize,transCrop, trBatchSize, trMaxEpoch, learningRate, timestampLaunch, nnArchitecture['ckpt'])


#--------------------------------------------------------------------------------

def runTest():

    trBatchSize = 24
    imgtransResize = 256
    imgtransCrop = 224

    print ('Testing the trained model')
    pathModel='../models/model-m-11072018-225455-DenseNet201_loss = 0.001958067467107384_acc = 0.994636148354_best_acc.pth.tar'
    # micro 3
    # pathcsvData = ['/media/bmi/Varghese1/bla/cataract-2018-test/micro_3/test11/',
    #                 '/media/bmi/Varghese1/bla/cataract-2018-test/micro_3/test12/',
    #                 '/media/bmi/Varghese1/bla/cataract-2018-test/micro_3/test13/',
    #                 '/media/bmi/Varghese1/bla/cataract-2018-test/micro_3/test14/',
    #                 '/media/bmi/Varghese1/bla/cataract-2018-test/micro_3/test15/']

    pathcsvData = [ #'/media/bmi/Varghese1/bla/cataract-2018-test/micro_2/test08/',
                    #'/media/bmi/Varghese1/bla/cataract-2018-test/micro_2/test09/',
                    #'/media/bmi/Varghese1/bla/cataract-2018-test/micro_2/test10/',
                    #'/media/bmi/Varghese1/bla/cataract-2018-test/micro_4/test16/',
                    #'/media/bmi/Varghese1/bla/cataract-2018-test/micro_4/test17/',
                    #'/media/bmi/Varghese1/bla/cataract-2018-test/micro_4/test18/',
                    #'/media/bmi/Varghese1/bla/cataract-2018-test/micro_4/test19/',
                    #'/media/bmi/Varghese1/bla/cataract-2018-test/micro_4/test20/',
                    #'/media/bmi/Varghese1/bla/cataract-2018-test/micro_5/test21/',
                    #'/media/bmi/Varghese1/bla/cataract-2018-test/micro_5/test22/',
                    #'/media/bmi/Varghese1/bla/cataract-2018-test/micro_5/test23/',
                    #'/media/bmi/Varghese1/bla/cataract-2018-test/micro_5/test24/',
                    #'/media/bmi/Varghese1/bla/cataract-2018-test/micro_5/test25/']
                    '/media/bmi/Varghese1/bla/cataract-2018-test/micro_1/test01/',
                    '/media/bmi/Varghese1/bla/cataract-2018-test/micro_1/test02/',
                    '/media/bmi/Varghese1/bla/cataract-2018-test/micro_1/test03/',
                    '/media/bmi/Varghese1/bla/cataract-2018-test/micro_1/test04/',
                    '/media/bmi/Varghese1/bla/cataract-2018-test/micro_1/test05/']
    for path in pathcsvData:
        Trainer.infer(model = DenseNet201(classCount = nclasses, isTrained = True), pathcsvData=path, pathModel=pathModel, trBatchSize=trBatchSize, transResize=imgtransResize, transCrop=imgtransCrop)
#--------------------------------------------------------------------------------

if __name__ == '__main__':
    # main()
    print ('once here')
    runTest()
