""" module contains functions for data manipulation, preprocessing """  

import numpy as np
import pandas as pd 
import os
import module_featureExtraction as featureExtract


def printCSV(dataSubfolder, timestamps, inputFilePrefix, inputFeaturesForLogTransform, \
                        inputFeaturesToDrop, reactorDimensions, reactorElements): 
    """Print  formatted time data (csv)
    Args:
        dataSubfolder : string, subfolder name for printing csv
        timeStamps : list, time stamps to aggregate data on 
        inputFilePrefix : string, prefix of input files to be read 
        inputFeaturesForLogTransform : list, features to transform to log space 
        inputFeaturesToDrop : list, features to drop before printing to csv
        reactorDimensions : tuple, reactor diameter x reactor ht
        reactorElements : tuple, elements along diameter and height 
    Returns: None
    """
    # 1. feature extraction, 2. drop features, 3. write csv
    targetDirectory = os.getcwd() + '\\' + dataSubfolder
    for index, itime in enumerate(timestamps): 
        currenttimefile = inputFilePrefix + str(itime) + '.txt'
        # preprocessing 
        inputFeatureMatrix = featureExtract.input_features(currenttimefile, \
            reactorDimensions, reactorElements, inputFeaturesForLogTransform)     
        # drop features 
        if len(inputFeaturesToDrop)>0: 
            inputFeatureMatrix = inputFeatureMatrix.drop(inputFeaturesToDrop, axis=1)
        # write csv 
        inputFeatureMatrix.to_csv(targetDirectory+'\\'+inputFilePrefix+str(itime)+'.csv',index = False)
        # display progress 
        if index%10 == 0:
            print(str(index)+'/'+str(len(timestamps))+' files converted!')       
    return None
    

    
def writeFormattedData(dataDirectory, timeStampsAvailable, inputFilePrefix, \
                    inputFeaturesForLogTransform, inputFeaturesToDrop, \
                    reactorDimensions, reactorElements):
    """Aggregates timestamps, store csv in subfolder: formattedData within dataDirectory
    Args:
        dataDirectory : string, folder name for data name for printing csv
        timeStampsAvailable : integer, total time data in dataDirectory 
        inputFilePrefix : string, prefix of input files to be read 
        inputFeaturesForLogTransform : list, features to transform to log space 
        inputFeaturesToDrop : list, features to drop before printing to csv
        reactorDimensions : tuple, reactor diameter x reactor ht
        reactorElements : tuple, elements along diameter and height 
    Returns: None
    """
    abspath = os.getcwd()
    os.chdir(abspath + '\\' + dataDirectory)
    os.mkdir(os.getcwd() + '\\formattedData' )
    timestampsAll = np.arange(1,timeStampsAvailable)
    printCSV('formattedData', timestampsAll, inputFilePrefix, inputFeaturesForLogTransform, \
                        inputFeaturesToDrop, reactorDimensions, reactorElements)    
    os.chdir(abspath)    
    return None
                
    

def trainValidTestSplit(timeStampsAvailable, fractionTrainCVTest, randomize = 0): 
    """Aggregates timestamps, store csv in subfolder: formattedData within dataDirectory
    Args:
        timeStampsAvailable : integer, total time data in dataDirectory 
        fractionTrainCVTest : float, fraction of data for train, CV and test
        randomize : random sampling for train, valid and test data  
    Returns: 
        timestampsTrain : list, training time stamps 
        timestampsCV : list, validation time stamps 
        timestampsTest : list, test time stamps 
    """
    fractionTrainCVTest = fractionTrainCVTest/np.sum(fractionTrainCVTest) # normalize
    timestamps = np.arange(1,timeStampsAvailable-2)
    if randomize: 
        timestamps = np.random.permutation(timestamps)
    timestampsTrain = timestamps[:int(fractionTrainCVTest[0]*len(timestamps))]
    timestampsCV = timestamps[int(fractionTrainCVTest[0]*len(timestamps)):int((fractionTrainCVTest[0]+fractionTrainCVTest[1])*len(timestamps))]
    timestampsTest = timestamps[int((fractionTrainCVTest[0]+fractionTrainCVTest[1])*len(timestamps)):] 
    
    return timestampsTrain, timestampsCV, timestampsTest



def dataTransform(data, ny, nx, imageDimensions):
    """ transforms 2D matrix in format [xloc, yloc, feature1, feature2 .... ]
        to 3D matrix in format (nFeatures x nx x ny) 
    Args:
        data : dataframe of shape (#cells, #features) 
        nx : integer, # cells in x dimension
        ny : integer, # cells in y dimension 
        imageDimensions : tuple, dimensions of trimmed image (xmin, xmax, ymin, ymax) 
    Returns: 
        dataImage : matrix of shape (ny, nx, nFeatures) 
    """
    nFeatures = np.shape(data)[1]
    dataImage = np.zeros((ny, nx, nFeatures))
    for ind, col in enumerate(list(data)): 
      dataImage[:,:,ind]=np.flipud(data[col].values.reshape((ny, nx),order ='C'))   
    return dataImage[imageDimensions[2]:imageDimensions[3],imageDimensions[0]:imageDimensions[1],:]



def compileData(timestamps, dataDirectory, reactorElements, imageDimensions):
    """ compile training / valid / test data
    Args:
        timestamps : list, timestamps to aggregate data on 
        dataDirectory : string, location of data directory 
        reactorElements : tuple, elements along diameter and height 
        imageDimensions : tuple, dimensions of trimmed image (xmin, xmax, ymin, ymax) 
    Returns: 
        data_X : array, training x data of shape (#timestamps, ny, nx, nFeatures) 
        data_Y : array, training y data of shape (#timestamps, ny, nx, nFeatures) 
    """

    nx, ny = reactorElements[0], reactorElements[1] 
    allDataX, allDataY = [], []
    for index, itime in enumerate(timestamps): 
        
        dataXFile = pd.read_csv(dataDirectory+'\\formattedData\\time'+str(itime)+'.csv') # store input data    
        dataXFile = dataTransform(dataXFile, ny, nx, imageDimensions) # transform to images  
        allDataX.append(dataXFile)
 
        dataYFile = pd.read_csv(dataDirectory+'\\formattedData\\time'+str(itime+1)+'.csv') # store output data   
        dataYFile = dataTransform(dataYFile, ny, nx, imageDimensions) # transform to images     
        allDataY.append(dataYFile)

        if index%10 == 0: print(str(index)+'/'+str(len(timestamps))+' files added!') # progress 
    
    data_X = np.array(allDataX, dtype='float32')
    data_Y = np.array(allDataY, dtype='float32')
    
    return data_X, data_Y