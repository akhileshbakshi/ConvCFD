import numpy as np 
import matplotlib.pyplot as plt

# basic plots 
def plotHistogram(data, size=(1, 1)): 
    """plot output histogram 
    Args:
      data : array of size [ntime, nx, ny, nFeatures]  
    Returns: 
      None 
    """

    class features(object):
        def __init__(self, name): 
            self.name = name
            
    allFeatures = [features('voidage'), \
                   features('gas x-vel'), \
                   features('gas y-vel'), \
                   features('solids x-vel'), \
                   features('solids y-vel'), \
                   features('pressure'), \
                   features('theta'), \
                   features('wall distance')]
    
    labelfont = {'weight' : 'bold', 'size'   : 14}
    nFeatures = np.shape(data)[3] 
    for icol in range(nFeatures): 
        dataFlattened = data[:,:,:,icol].flatten()
        rangeMin = np.sort(dataFlattened)[int(len(dataFlattened)*0.005)]
        rangeMax = np.sort(dataFlattened)[int(len(dataFlattened)*0.995)]
        weights = np.ones_like(dataFlattened)/float(len(dataFlattened))
        fig = plt.figure()
        plt.hist(dataFlattened, 25, facecolor='g', alpha=0.75, 
                 weights=weights, range=[rangeMin, rangeMax]) 
        fig.set_size_inches(size)
        plt.xlabel(allFeatures[icol].name, **labelfont)
        plt.ylabel('PDF [-]', **labelfont)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.show()
    return None 


def plot2Dsurf(X, Y, Z, title, size=(1, 1), minColor = 0, maxColor = 1, steps = 0.2): 
    """plot 2D surface 
     Args:
         X : array, 2D meshgrid storing x locations 
         Y : array, 2D meshgrid storing y locations 
         Z : array, quantity to be plotted where Z(i,j) corresponds to (X(i,j), Y(i,j))
     Returns: 
         2D surface plot 
    """
    
    labelfont = {'weight' : 'bold', 'size'   : 14}
    titlefont = {'weight' : 'bold', 'size'   : 16}
    fig = plt.figure()
    fig.set_size_inches(size)
    pcolorplot = plt.pcolor(X, Y, Z, cmap='RdBu_r', vmin=minColor, vmax=maxColor)
    plt.axis([X.min(), X.max(), Y.min(), Y.max()])
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('x-location', **labelfont)
    plt.ylabel('y-location', **labelfont)
    plt.title(title, **titlefont)  
    plt.xticks([0, 16, 33, 50, 66])
    plt.yticks([0, 20, 40, 60, 80])
    plt.colorbar(pcolorplot, shrink=0.5, aspect=5, boundaries=np.linspace(minColor,maxColor,11), 
                 ticks = np.arange(minColor, maxColor+steps, steps))
    return fig


def plotScatter(X, Y, title, size = (1,1)): 
    """plot scatter plot
    Args:
      X : list
      Y : list 
      title : str
      size : tuple 
    Returns: 
      None
    """
    minX, maxX = np.min(X), np.max(X)
    labelfont = {'weight' : 'bold', 'size'   : 12}
    titlefont = {'weight' : 'bold', 'size'   : 16}    
    fig = plt.figure()
    fig.set_size_inches(size)
    plt.scatter(X, Y) 
    plt.plot(np.arange(minX, maxX), np.arange(minX, maxX))
    plt.xlabel('actual output', **labelfont)
    plt.ylabel('predicted output', **labelfont)
    plt.axis([minX, maxX, minX, maxX])
    plt.title(title, **titlefont)    
    plt.show() 


def plotFieldVariables(model, xData, yData, sigma, mu, imageDimensions, \
                       rangeFeatures, timeSeed = 1, savefig = 0): 
    """plots input, true output and predicted output using CNN specified by 'model'
    Args:
      model : keras CNN model
      xData : array, input data of size [ntimes, nx, ny, nFeatures]
      yData : array, output data (actual) of size [ntimes, nx, ny, nFeatures]
      sigma : list, range / standard deviation of length [nfeatures] 
      mu : list, min / mean of length [nfeatures] 
      imageDimensions : tuple, image extremities 9xmin, xmax, ymin, ymax) 
      timeSeed : int, time seed for display 
      savefig : bool, 1: save figure
      rangeFeatures : list of features to be considered
    Returns: 
      None
    """

    class features(object):
        def __init__(self, name, minValue, maxValue, steps): 
            self.name = name
            self.minValue = minValue 
            self.maxValue = maxValue 
            self.steps = steps 
            
    allFeatures = [features('voidage', 0.4, 1, 0.2), \
                   features('gas x-vel', -0.6, 0.6, 0.3), \
                   features('gas y-vel', 0.2, 4.2, 1.0), \
                   features('solids x-vel', -0.4, 0.4, 0.2), \
                   features('solids y-vel', -0.4, 0.4, 0.2), \
                   features('pressure', 0, 20000, 5000), \
                   features('theta', -5, 3, 2)]
    
    Xmin, Xmax, Ymin, Ymax = imageDimensions
    X, Y = np.meshgrid(np.arange(Xmin,Xmax), np.arange(Ymax,Ymin,-1))
    
    for f in rangeFeatures: 
        actualInput = (xData[timeSeed,:,:,f] * sigma[f]) + mu[f]
        actualOutput = (yData[timeSeed,:,:,f] * sigma[f]) + mu[f]
        predictedOutput = (model.predict(xData[timeSeed:timeSeed+1,:,:,:])[0,:,:,f] * sigma[f]) + mu[f]
        
        # input, true output, predicted output  
        fig1 = plot2Dsurf(X, Y, actualInput, allFeatures[f].name + ' input', size=(4.2,4), \
                         minColor = allFeatures[f].minValue, 
                         maxColor = allFeatures[f].maxValue, 
                         steps = allFeatures[f].steps) 
        fig2 = plot2Dsurf(X, Y, actualOutput, allFeatures[f].name + ' output', size=(4.2,4), \
                         minColor = allFeatures[f].minValue, 
                         maxColor = allFeatures[f].maxValue, 
                         steps = allFeatures[f].steps) 
        fig3 = plot2Dsurf(X, Y, predictedOutput, allFeatures[f].name + ' predicted', size=(4.2,4), \
                         minColor = allFeatures[f].minValue, 
                         maxColor = allFeatures[f].maxValue, 
                         steps = allFeatures[f].steps) 
        if savefig: 
            fig1.savefig('time' + str(timeSeed) + '_' + allFeatures[f].name + 'Input.png') 
            fig2.savefig('time' + str(timeSeed) + '_' + allFeatures[f].name + 'Output.png')
            fig3.savefig('time' + str(timeSeed) + '_' + allFeatures[f].name + 'Predicted.png')    
    return None 


def evaluateError(model, xData, yData, sigma, mu, rangeFeatures): 
    """print massError, MSE and accuracy for selected features 
    Args:
      model : keras CNN model
      xData : array, input data of size [ntimes, nx, ny, nFeatures]
      yData : array, output data (actual) of size [ntimes, nx, ny, nFeatures]
      sigma : list, range / standard deviation of length [nfeatures] 
      mu : list, min / mean of length [nfeatures] 
      rangeFeatures : list of features to be considered 
    Returns: 
      None
    """    
       
    # compute mass error 
    f, sumError = 0, 0
    for timeSeed in range(np.shape(xData)[0]): 
        actualOutput = (yData[timeSeed,:,:,f] * sigma[f]) + mu[f]
        predictedOutput = (model.predict(xData[timeSeed:timeSeed+1,:,:,:])[0,:,:,f] * sigma[f]) + mu[f]
        error = (np.sum(1-actualOutput) - np.sum(1-predictedOutput))/np.sum(1-actualOutput) 
        sumError += error
    massErrorAvg = sumError / np.shape(xData)[0]      
    print('mass error [%] = ' + str(np.round(massErrorAvg*100,2)))

    # compute MSE  
    sumAllError = 0 
    for index, f in enumerate(rangeFeatures):
        sumError, elementCount = 0, 0     
        for timeSeed in range(np.shape(xData)[0]): 
            actualOutput = (yData[timeSeed,:,:,f] * sigma[f]) + mu[f]
            predictedOutput = (model.predict(xData[timeSeed:timeSeed+1,:,:,:])[0,:,:,f] * sigma[f]) + mu[f]  
            error = np.sum((actualOutput - predictedOutput)**2)
            sumError += error
            elementCount += np.shape(xData)[1] * np.shape(xData)[2]
        meanError = sumError / elementCount 
        sumAllError += meanError 
        print('feature ' + str(f) + ' MSE = ' + str(np.round(meanError,4)))
    meanAllError = sumAllError / len(rangeFeatures)
    print('All MSE = ' + str(np.round(meanAllError,4)))

    # compute accuracy  
    SSres, SStot = 0, 0 
    for index, f in enumerate(rangeFeatures):
        sumError, elementCount = 0, 0     
        for timeSeed in range(np.shape(xData)[0]): 
            actualOutput = (yData[timeSeed,:,:,f] * sigma[f]) + mu[f]
            predictedOutput = (model.predict(xData[timeSeed:timeSeed+1,:,:,:])[0,:,:,f] * sigma[f]) + mu[f]  
            SSres += np.sum((actualOutput - predictedOutput)**2)
            SStot += np.sum((actualOutput - np.mean(actualOutput))**2)
        accuracy = 1 - SSres / SStot
        print('feature ' + str(f) + ' accuracy = ' + str(np.round(accuracy,4)))
    return None 

   
def errorScatter(model, xData, yData, sigma, mu, rangeFeatures):
    """plots error scatters for actual output and predicted output of features
    Args:
      model : keras CNN model
      xData : array, input data of size [ntimes, nx, ny, nFeatures]
      yData : array, output data (actual) of size [ntimes, nx, ny, nFeatures]
      sigma : list, range / standard deviation of length [nfeatures] 
      mu : list, min / mean of length [nfeatures] 
      rangeFeatures : list of features to be considered
    Returns: 
      None
    """    

    class features(object):
        def __init__(self, name, minValue, maxValue): 
            self.name = name
            self.minValue = minValue 
            self.maxValue = maxValue 
            
    allFeatures = [features('void', 0.4, 1), \
                   features('gas x-vel', -0.6, 0.6), \
                   features('gas y-vel', 0, 4.5), \
                   features('solids x-vel', -0.6, 0.4), \
                   features('solids y-vel', -0.4, 0.4), \
                   features('pressure', 0, 20000), \
                   features('theta', -5, 3)]                  

    for f in rangeFeatures:    
        allXpoints, allYpoints = [], []
        for timeSeed in range(np.shape(xData)[0]): 
            actualInput = (xData[timeSeed,:,:,f] * sigma[f]) + mu[f]
            actualOutput = (yData[timeSeed,:,:,f] * sigma[f]) + mu[f]
            predictedOutput = (model.predict(xData[timeSeed:timeSeed+1,:,:,:])[0,:,:,f] * sigma[f]) + mu[f]  
            allXpoints.append(actualOutput.flatten())
            allYpoints.append(predictedOutput.flatten())
        plotScatter(allXpoints, allYpoints, allFeatures[f].name, size = (4,4))   
    return None 

           

