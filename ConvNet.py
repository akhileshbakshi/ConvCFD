# setup workspace ------------------------------------------------------------- 
from IPython import get_ipython
def __reset__(): get_ipython().magic('reset -sf')   # reset all 
__reset__()

import os
abspath = 'D:\\ConvCFD' 
os.chdir(abspath)

# import modules --------------------------------------------------------------
import numpy as np
import keras.backend as K
K.set_image_data_format('channels_last')
# user-defined modules 
import module_filePreprocessing as filePre
import module_convNetArchitecture as convNetArch
import module_dataVisualization as dataVis
import module_dataPipeline as dataPip


# input parameters ------------------------------------------------------------ 

# data aggregation 
dataDirectory = 'simulation_data_sample'# folder with simulation files 
fileprefix = 'time'                     # filenames: fileprefix+<file #>+'.txt' 
timeStampsAvailable = 500               # total timeframes inside dataDirectory
fractionTrainCVTest = [0.9, 0.0, 0.1]   # fraction split into train, CV and test
                
# input-output 
inputFeaturesForLogTransform = ['theta'] 
inputFeaturesToDrop =  ['xloc', 'yloc']  
    
# geometry mapping 
reactorDimensions = (20, 24)            # reactor diameter x reactor ht [cm] 
reactorElements = (66, 80)              # # elements along diameter, height 


    
# create formatted data -------------------------------------------------------
# write formatted data into 'formattedData' subfolder within dataDirectory
# needs to be invoked only once to write data in requisite format 
filePre.writeFormattedData(dataDirectory, timeStampsAvailable, fileprefix, 
    inputFeaturesForLogTransform, inputFeaturesToDrop, reactorDimensions, reactorElements)

# create training, valid data -------------------------------------------------
# timestamps are arrays of time frames aggregated in train, validation, test data 
fractionTrainCVTest = np.array(fractionTrainCVTest)
timestampsTrain, timestampsCV, timestampsTest = \
    filePre.trainValidTestSplit(timeStampsAvailable, fractionTrainCVTest, randomize = 1)

# aggregate data in image format ----------------------------------------------
# train_X of shape (#timestampsTrain, ny, nx, nFeatures) 
imageDimensions = (0, 66, 0, 80) 
train_X, train_Y = filePre.compileData(timestampsTrain, dataDirectory, reactorElements, imageDimensions)
test_X, test_Y = filePre.compileData(timestampsTest, dataDirectory, reactorElements, imageDimensions)



# work with data-copies -------------------------------------------------------
    
train_X1, test_X1 = np.copy(train_X), np.copy(test_X)
train_Y1, test_Y1 = np.copy(train_Y), np.copy(test_Y)
train_Y1, test_Y1 = train_Y1[:,:,:,:5], test_Y1[:,:,:,:5]   

# data has format [nfiles x nx x ny x nfeatures]
# X has nfeatures = [epg, ug, vg, us, vs, pressure, theta, wall-distance]  
# Y has nfeatures = [epg, ug, vg, us, vs]

# scale features based on 2%ile, 98% points of distribution 
nFeatures = np.shape(train_X1)[3]
mu, sigma = np.zeros((nFeatures,1)), np.zeros((nFeatures,1))
for icol in range(nFeatures):
    temp = train_X1[:,:,:,icol].flatten()
    m1 = np.sort(temp)[int(len(temp)*0.02)] # 2%ile 
    m2 = np.sort(temp)[int(len(temp)*0.98)] # 98%ile 
    mu[icol], sigma[icol] = m1, m2 - m1


# creating classes based on voidage (epg) 
class defineClass(object): 
    def __init__(self, minepg, maxepg): 
        self.minepg = minepg 
        self.maxepg = maxepg
      
lrgnum = 100  # use for one sided constraint     
allClasses = [defineClass(-lrgnum, 0.42), \
              defineClass(0.42, 0.6), \
              defineClass(0.6, 0.95), \
              defineClass(0.95, lrgnum)]

# computing class frequency, weights 
numClass = len(allClasses)
epgFlat = train_Y1[:,:,:,0].flatten()  
weight, numInstances = np.zeros((numClass,1)), np.zeros((numClass,1))
for iClass in range(numClass): 
    minepg, maxepg = allClasses[iClass].minepg, allClasses[iClass].maxepg
    numInstances[iClass] = np.sum((epgFlat >= minepg) & (epgFlat < maxepg))
numInstancesMax = max(numInstances) 
weight = numInstancesMax / numInstances






# model training --------------------------------------------------------------
  
# references for choices 
# https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.57.5612&rep=rep1&type=pdf
# https://arxiv.org/pdf/1609.04836.pdf
# http://ruder.io/optimizing-gradient-descent/    
# need to optimize learning rate schedules- 
# https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1       


train_X1 = dataPip.allOperations(train_X1, mu, sigma, allClasses, weight, appendClass = 0)
train_Y1 = dataPip.allOperations(train_Y1, mu, sigma, allClasses, weight, appendClass = 1)
dataVis.plotHistogram(train_X1, size = (4,4))
test_X1 = dataPip.allOperations(test_X1, mu, sigma, allClasses, weight, appendClass = 0)
test_Y1 = dataPip.allOperations(test_Y1, mu, sigma, allClasses, weight, appendClass = 1) 
      
nKernels = [5, 5] 
nFilters = [32, 16]
nLayers =  [2, 2]
nFeaturesIn, nFeaturesOut = np.shape(train_X1)[3],  np.shape(train_Y1)[3]
model = convNetArch.ConvCFD(nGridCellsX = 66, nGridCellsY = 80, \
                    nFeatures = nFeaturesIn, nFeaturesOut = nFeaturesOut, \
                    kernelRegularizer = 0.0, biasRegularlizer = 0.0, \
                    nKernels = nKernels, nLayers = nLayers, nFilters = nFilters)
trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
print('# trainable parameters = ' + str(trainable_count))

[model, Model] = convNetArch.trainNetwork(train_X1, train_Y1, model, lamda1 = 1, \
             epochs = 1000, learnRate = 0.1, batchSize = 20, decayRate = 5*10**-7)


# output statistics -----------------------------------------------------------
rangeFeatures = [0,1,2,3,4] 
dataVis.evaluateError(model, test_X1, test_Y1, sigma, mu, rangeFeatures)
dataVis.errorScatter(model, test_X1, test_Y1, sigma, mu, rangeFeatures)
timeSeed = 2 
dataVis.plotFieldVariables(model, test_X1, test_Y1, sigma, mu, imageDimensions, 
                           rangeFeatures, timeSeed, savefig = 1)


#save model -------------------------------------------------------------------
filename = 'model_mu-sigma.txt'; file = open(filename, 'w')
file.write("%i\n" % nFeatures)
for item in mu: file.write("%f\n" % item)
for item in sigma: file.write("%f\n" % item)
file.close()

filename = 'model_weight.txt'; file = open(filename, 'w')
file.write("%i\n" % numClass)
for item in weight: file.write("%f\n" % item)
file.close()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_weights")
print("Saved model to disk")




