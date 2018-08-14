""" module contains functions for feature extraction """ 

import numpy as np
import pandas as pd

def input_features(currentTimeFile, reactorDimensions, reactorElements, inputFeaturesForLogTransform): 
    """ extract input features from simulation output 
    Args:
        currenttimefile : string, file name to extract features from 
        inputFeaturesForLogTransform : list, features to transform to log space 
        reactorDimensions : tuple, reactor diameter x reactor ht
        reactorElements : tuple, elements along diameter and height 
    Returns: 
        data : dataFrame with all features and size (# cells, # features)
    """   
    reactorDia, reactorHt = reactorDimensions[0], reactorDimensions[1] 
    simElementsDia, simElementsHt = reactorElements[0], reactorElements[1] 

    # read data 
    data = pd.read_csv(currentTimeFile, sep=",", header = None)
    data.columns = ["cellctr", "epg", "Pg", "ug", "vg", "us", "vs", "theta"]    

    # setup geometry     
    dx, dy = reactorDia/simElementsDia, reactorHt/simElementsHt 
    nxelements, nyelements = simElementsDia, simElementsHt 

    # convert cellctr to xloc and yloc, scale by bed diameter  
    yloc = np.floor_divide(data.cellctr.values,nxelements)
    yloc[nxelements-1::nxelements] -= 1 
    xloc = data.cellctr.values - yloc*nxelements
    yloc, xloc = (yloc +0.5) * dy/reactorDia, (xloc -0.5) * dx/reactorDia  
    data["xloc"], data["yloc"] = pd.DataFrame(data = xloc), pd.DataFrame(data = yloc)
    data = data.drop(["cellctr"], axis=1)
    
    # convert variables to equivalent form as in mass, momentum and theta equations 
    data["epgug"] = data.epg * data.ug 
    data["epgvg"] = data.epg * data.vg 
    data["epsus"] = (1-data.epg) * data.us 
    data["epsvs"] = (1-data.epg) * data.vs
    data["theta"] = data.theta**0.5 
    
    #drop irrelevant columns and rearrange 
    data = data.drop(["ug", "vg", "us", "vs"], axis=1)
    data = data[["xloc", "yloc", "epg", "epgug", "epgvg", "epsus", "epsvs", "Pg", "theta"]] 
    
    # take log transforms
    if len(inputFeaturesForLogTransform)>0: 
        data[inputFeaturesForLogTransform] = np.log(data[inputFeaturesForLogTransform])
     
    walldistance = np.minimum(data.xloc.values, 1 - data.xloc.values)   
    data['walldistance'] = pd.DataFrame(data = walldistance)
  
    return data 
    
    

            
            
                         
        
    
    