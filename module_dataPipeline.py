import numpy as np 

def allOperations(data, mu, sigma, allClasses, weight, appendClass = 0): 
    """ scale data and add classes 
    Args:
        data : numpy array [nframes, nx, ny, nfeatures], input array 
        mu : list of shape (nfeatures,1), mean or parameter for scaling 
        sigma : list of shape (nfeatures,1), stdev or parameter for scaling 
        allClasses : list, (min, max) for epg 
        weight : list, weights for classes 
        appendClass : boolean, add weight to data array  
    Returns: 
        data  
    """     
    
    numFeatures = min(len(mu), np.shape(data)[3])
    # scale features based on mu and sigma
    for icol in range(numFeatures):  
        data[:,:,:,icol] = (data[:,:,:,icol] - mu[icol]) / sigma[icol]
    # appending weights to output    
    if appendClass: 
        numClass = len(allClasses)
        dataClass = np.zeros(np.shape(data[:,:,:,0:1]))
        epgdata = data[:,:,:,0:1]            
        for iClass in range(numClass): 
            minepg, maxepg = allClasses[iClass].minepg, allClasses[iClass].maxepg            
            minepg, maxepg = (minepg - mu[0])/sigma[0], (maxepg - mu[0])/sigma[0]
            dataClass[(epgdata >= minepg) & (epgdata < maxepg)] = weight[iClass]                      
        data = np.append(data, dataClass, axis = 3)  
    return data 


def allOperationsInverse(data, mu, sigma, discardClass = 0): 
    """ reverses allOperations 
    Args:
        data : numpy array [nframes, nx, ny, nfeatures], input array 
        mu : list of shape (nfeatures,1), mean or parameter for scaling 
        sigma : list of shape (nfeatures,1), stdev or parameter for scaling 
        discardClass : boolean, discard weight from data array  
    Returns: 
        data      
    """
    if discardClass: data = data[:,:,:,:-1]
    # recover original range of features 
    numFeatures = min(len(mu), np.shape(data)[3])
    for icol in range(numFeatures): 
        data[:,:,:,icol] = (data[:,:,:,icol] * sigma[icol]) + mu[icol] 

    return data 
        
