import os
import numpy as np
import time
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func
from torch.utils.data import DataLoader

from sklearn.metrics.ranking import roc_auc_score


from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchnet as tnt
import pandas as pd
from GLN_DataGenerator import  DatasetGenerator

nclasses = 21
confusion_meter = tnt.meter.ConfusionMeter(nclasses, normalized=True)
from tqdm import tqdm
from torchvision import transforms
#--------------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer ():
    #---- Train the densenet network
    #---- TrainVolPaths - path to the directory that contains images
    #---- TrainLabels - path to the file that contains image paths and label pairs (training set)
    #---- ValidVolPaths - path to the directory that contains images
    #---- ValidLabels - path to the file that contains image paths and label pairs (training set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnClassCount - number of output classes
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file

    #---- TODO:
    #---- checkpoint - if not None loads the model and continues training

    def train (self, TraincsvPath, ValidcsvPath, nnArchitecture,
                nnClassCount,transResize,transCrop, trBatchSize,
                trMaxEpoch, learningRate, timestampLaunch, checkpoint):

        # print ("++++++++++++++++++++++++++++++++++++++++++++++++", device)
        #-------------------- SETTINGS: NETWORK ARCHITECTURE
        model = nnArchitecture['model'].to(device)
        # print (next(model.parameters()).is_cuda)
        # model = torch.nn.DataParallel(model)


        transResize= 256

        transCrop  =512
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.CenterCrop(transCrop))
        transformList.append(transforms.Resize([transResize,transResize]))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        transformSequence1=transforms.Compose(transformList)

        transCrop  = 1024
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



        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam (model.parameters(), lr=learningRate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)## based on Zoozog
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')

        #-------------------- SETTINGS: LOSS
        loss = torch.nn.BCELoss()

        lossMIN = 100000
        accMax  =0
        #---- Load checkpoint
        if checkpoint != None:
            saved_parms=torch.load(checkpoint)
            model.load_state_dict(saved_parms['state_dict'])
            optimizer.load_state_dict(saved_parms['optimizer'])
            start_epoch= saved_parms['epochID']
            lossMIN    = saved_parms['best_loss']
            accMax     =saved_parms['best_acc']
            print ("===> model loaded successfully!!....")
            # print (saved_parms['confusion_matrix'])

        #---- TRAIN THE NETWORK

        lossMIN = 100000
        accMax  = 0
        sub = pd.DataFrame()

        timestamps = []
        archs = []
        losses = []
        accs  = []
        epochs = []


        for epochID in range (0, trMaxEpoch):
        #-------------------- SETTINGS: DATASET BUILDERS
            # np.random.shuffle(TraincsvPath)
            datasetTrain = DatasetGenerator(CSV_PATH = TraincsvPath,
                                            transforms1 = transformSequence1,
                                            transforms2 = transformSequence2,
                                            transforms3 = transformSequence3)

            datasetVal =   DatasetGenerator(CSV_PATH =ValidcsvPath,
                                            transforms1 = transformSequence1,
                                            transforms2 = transformSequence2,
                                            transforms3 = transformSequence3)

            dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize,
                                        shuffle=True, num_workers=8, pin_memory=False) ## Shuffling inside dataloader  using np
            dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize,
                                        shuffle=True, num_workers=8, pin_memory=False)

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime

            print (str(epochID)+"/" + str(trMaxEpoch) + "---")
            self.epochTrain (model, dataLoaderTrain, optimizer, scheduler, trMaxEpoch, nnClassCount, loss, trBatchSize)
            lossVal, losstensor, _cm = self.epochVal (model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, nnClassCount, loss, trBatchSize)

            currAcc = float(np.sum(np.eye(nclasses)*_cm.conf))/np.sum(_cm.conf)
            # print (_cm.conf)

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            launchTimestamp = timestampDate + '-' + timestampTime

            scheduler.step(losstensor.item())

            if lossVal < lossMIN:
                lossMIN = lossVal

                timestamps.append(launchTimestamp)
                archs.append(nnArchitecture['name'])
                losses.append(lossVal)
                accs.append(currAcc)
                epochs.append(epochID)

                model_name = '../models/model-m-' + launchTimestamp + "-" + nnArchitecture['name'] + '_loss = ' + str(lossVal) + '_acc = '+str(currAcc) + '_best_loss.pth.tar'

                states = {'epochID': epochID + 1,
                            'arch': nnArchitecture['name'],
                            'state_dict': model.state_dict(),
                            'best_acc': currAcc,
                            'confusion_matrix':_cm.conf,
                            'best_loss':lossMIN,
                            'optimizer' : optimizer.state_dict()}

                torch.save(states, model_name)
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + launchTimestamp + '] loss= ' + str(lossVal) +' acc= '+str(currAcc))

            elif currAcc > accMax:
                accMax  = currAcc

                timestamps.append(launchTimestamp)
                archs.append(nnArchitecture['name'])
                losses.append(lossVal)
                accs.append(currAcc)
                epochs.append(epochID)

                model_name = '../models/model-m-' + launchTimestamp + "-" + nnArchitecture['name'] + '_loss = ' + str(lossVal) + '_acc = '+str(currAcc) + '_best_acc.pth.tar'

                states = {'epochID': epochID + 1,
                            'arch': nnArchitecture['name'],
                            'state_dict': model.state_dict(),
                            'best_acc': accMax,
                            'confusion_matrix':_cm.conf,
                            'best_loss':lossVal,
                            'optimizer' : optimizer.state_dict()}

                torch.save(states, model_name)
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + launchTimestamp + '] loss= ' + str(lossVal) +' acc= '+str(currAcc))

            else:
                print ('Epoch [' + str(epochID + 1) + '] [---] [' + launchTimestamp + '] loss= ' + str(lossVal) +' acc= '+str(currAcc))


        sub['epoch']  = epochs
        sub['timestamp'] = timestamps
        sub['archs'] = archs
        sub['loss'] = losses
        sub['Acc']  = accs


        sub.to_csv('../models/' + nnArchitecture['name'] + '.csv', index=True)


    def epochTrain (self, model, dataLoader, optimizer, scheduler, epochMax, classCount, loss, trBatchSize):

        phase='train'
        with torch.set_grad_enabled(phase == 'train'):
            for batchID, (image, gt, _) in tqdm(enumerate (dataLoader)):
                target = gt
                varInputHigh = image.to(device)
                varTarget    = target.to(device)
                varOutput = model(varInputHigh)
                lossvalue = loss(varOutput, varTarget)
                optimizer.zero_grad()
                lossvalue.backward()
                optimizer.step()

    #--------------------------------------------------------------------------------

    def epochVal (self, model, dataLoader, optimizer, scheduler, epochMax, classCount, loss, trBatchSize):

        model.eval ()

        lossVal = 0
        lossValNorm = 0

        losstensorMean = 0
        confusion_meter.reset()


        with torch.no_grad():
            for i, (image, gt, _) in enumerate (dataLoader):
                target = gt
                varInput = image.to(device)
                varTarget    = target.to(device)
                varOutput = model(varInput)
                losstensor = loss(varOutput, varTarget)
                losstensorMean += losstensor
                confusion_meter.add(varOutput.view(-1), varTarget.data.view(-1))
                lossVal += losstensor.item()
                del losstensor
                del varOutput, varTarget, varInput
                lossValNorm += 1


            outLoss = lossVal / lossValNorm
            losstensorMean = losstensorMean / lossValNorm

        return outLoss, losstensorMean, confusion_meter

    #--------------------------------------------------------------------------------

    #---- Computes area under ROC curve
    #---- dataGT - ground truth data
    #---- dataPRED - predicted data
    #---- classCount - number of classes

    def computeAUROC (self,dataGT, dataPRED, classCount):

        outAUROC = []

        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()

        for i in range(classCount):
            try:
                outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            except:
                outAUROC.append(0)

        return outAUROC


    #--------------------------------------------------------------------------------

    #---- Test the trained network
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training

    def test (self, model, pathcsvData, pathModel, trBatchSize, transResize, transCrop):

        print (pathcsvData)

        CLASS_NAMES = ['biomarker', 'Charleux cannula',    'hydrodissection cannula', 'Rycroft cannula', 'viscoelastic cannula', 'cotton',  \
                    'capsulorhexis cystotome', 'Bonn forceps',    'capsulorhexis forceps',   'Troutman forceps',    'needle holder',   \
                    'irrigation/aspiration handpiece', 'phacoemulsifier handpiece',   'vitrectomy handpiece',    'implant injector',    \
                    'primary incision knife',  'secondary incision knife',    'micromanipulator'    ,'suture needle' ,  'Mendez ring', 'Vannas scissors']
        nnClassCount=len(CLASS_NAMES)

        cudnn.benchmark = True

        #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        # if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        # elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        # elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        saved_params = torch.load(pathModel)
        model.load_state_dict(saved_params['state_dict'])
        model = model.to(device)
        # model = torch.nn.DataParallel(model).cuda()

        # modelCheckpoint = torch.load(pathModel)
        # model.load_state_dict(modelCheckpoint['state_dict'])

        #-------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        #-------------------- SETTINGS: DATASET BUILDERS
        transResize= 256

        transCrop  =512
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.CenterCrop(transCrop))
        transformList.append(transforms.Resize([transResize,transResize]))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        transformSequence1=transforms.Compose(transformList)

        transCrop  = 1024
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

#        transformList = []
#        transformList.append(transforms.Resize([transResize,transResize]))
#        transformList.append(transforms.TenCrop(transCrop))
#        transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
#        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
#        transformSequence=transforms.Compose(transformList)


        datasetTest =   DatasetGenerator(CSV_PATH = pathcsvData,
                                        transforms1 = transformSequence1,
                                        transforms2 = transformSequence2,
                                        transforms3 = transformSequence3)

        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=8, shuffle=False, pin_memory=True)

        outGT = torch.FloatTensor().to(device)
        outPRED = torch.FloatTensor().to(device)

        model.eval()

        with torch.no_grad():

            for i, (input, target, _) in tqdm(enumerate(dataLoaderTest)):

                target = target.to(device)
                outGT = torch.cat((outGT, target), 0)

                bs, c, h, w = input.size()

                varInput = input.view(-1, c, h, w).to(device)

                out = model(varInput)
                # outMean = out.view(bs, n_crops, -1).mean(1)
                # print (outGT.size(),)
                outPRED = torch.cat((outPRED, out.data), 0)
            print (outGT.size(),outPRED.size())
            aurocIndividual = self.computeAUROC(outGT, outPRED, nnClassCount)
            aurocMean = np.array(aurocIndividual).mean()

            print ('AUROC mean ', aurocMean)

            for i in range (0, len(aurocIndividual)):
                print (CLASS_NAMES[i], ' ', aurocIndividual[i])


        return
#--------------------------------------------------------------------------------
    
    def infer (self, model, pathcsvData, pathModel, trBatchSize, transResize, transCrop):

        print (pathcsvData)

        CLASS_NAMES = ['biomarker', 'Charleux cannula',    'hydrodissection cannula', 'Rycroft cannula', 'viscoelastic cannula', 'cotton',  \
                    'capsulorhexis cystotome', 'Bonn forceps',    'capsulorhexis forceps',   'Troutman forceps',    'needle holder',   \
                    'irrigation/aspiration handpiece', 'phacoemulsifier handpiece',   'vitrectomy handpiece',    'implant injector',    \
                    'primary incision knife',  'secondary incision knife',    'micromanipulator'    ,'suture needle' ,  'Mendez ring', 'Vannas scissors']
        nnClassCount=len(CLASS_NAMES)

        cudnn.benchmark = True

        #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        # if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        # elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        # elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        saved_params = torch.load(pathModel)
        model.load_state_dict(saved_params['state_dict'])
        model = model.to(device)
        # model = torch.nn.DataParallel(model).cuda()

        # modelCheckpoint = torch.load(pathModel)
        # model.load_state_dict(modelCheckpoint['state_dict'])

        #-------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        #-------------------- SETTINGS: DATASET BUILDERS
        transResize= 256

        transCrop  =512
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.CenterCrop(transCrop))
        transformList.append(transforms.Resize([transResize,transResize]))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        transformSequence1=transforms.Compose(transformList)

        transCrop  = 1024
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

        datasetTest =   DatasetGenerator(CSV_PATH = pathcsvData,
                                        transforms1 = transformSequence1,
                                        transforms2 = transformSequence2,
                                        transforms3 = transformSequence3)

        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=8, shuffle=False, pin_memory=True)

        # outGT = torch.FloatTensor().to(device)
        outPRED = torch.FloatTensor().to(device)

        model.eval()
        Ids = []
        with torch.no_grad():

            for i, (input, _, path) in tqdm(enumerate(dataLoaderTest)):

                # target = target.to(device)
                # outGT = torch.cat((outGT, target), 0)
                # print (path)
                bs, c, h, w = input.size()

                varInput = input.view(-1, c, h, w).to(device)

                out = model(varInput)
                # outMean = out.view(bs, n_crops, -1).mean(1)
                # print (outGT.size(),)
                Ids    = np.concatenate((Ids, path), 0)
                outPRED = torch.cat((outPRED, out.data), 0)
            outPRED = outPRED.cpu().numpy()
            print (Ids.shape, outPRED.shape)
            result_data = pd.DataFrame(np.concatenate([np.expand_dims(Ids, 1), outPRED], axis=1))
            result_data.to_csv('../../../' + pathcsvData.split('/')[-2] + '.csv', header=None, index=False)

            # aurocIndividual = self.computeAUROC(outGT, outPRED, nnClassCount)
            # aurocMean = np.array(aurocIndividual).mean()

            # print ('AUROC mean ', aurocMean)

            #for i in range (0, len(aurocIndividual)):
                #print (CLASS_NAMES[i], ' ', aurocIndividual[i])


        return
