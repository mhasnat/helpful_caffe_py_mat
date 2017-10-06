# Function for computing:
# Cosine distance (given two sets of observations) and 
# Accuracy (given label and distances)
import numpy as np

                  
def getCosineDistance(data_left, data_right):
    # Normalize feature vectors
    
    data_left_n2 = data_left / np.sqrt(np.sum(data_left ** 2, axis=1, keepdims=True))
    data_right_n2 = data_right / np.sqrt(np.sum(data_right ** 2, axis=1, keepdims=True))
    
    # Get Cosine Distance
    dist_cos = 1 - np.sum(np.multiply(data_left_n2, data_right_n2), axis=1, keepdims=True) 
    
    return dist_cos

def computeAccuracy(dist_cos, labels):
    ## Cosine distance based accuracy on all test samples
    thDIst = np.arange(0.1, 5, 0.01)
    acc = np.zeros((len(thDIst)))
    for i in range(1,len(thDIst)):
        tlab = np.ones((len(labels)))
        tlab[np.where(dist_cos>=thDIst[i])[0]] = 0
        missCl = labels - tlab
        acc[i] = float(len(np.where(missCl==0)[0]))/len(labels)
    
    finalAcc = np.max(acc)
    finThVal = thDIst[np.argmax(acc)]
    
    return finalAcc, finThVal
