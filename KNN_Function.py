#==============================================================================
# KNN Function
# Created By Jan De Pue (2018) at Ghent University
# Reference:
#==============================================================================


import numpy
## GENERIC KNN FUNCTION FILE!

def KNN(Train_Pred,In_Pred,Train_Est,LeaveOneOut=False,Knumber=-1, Power = -1):
    
    nTrain=numpy.shape(Train_Pred)[0]
    nIn=numpy.shape(In_Pred)[0]
    nPar=numpy.size(Train_Pred,axis=1)
    nEst=numpy.size(Train_Est,axis=1)
    
    ## Rescale
    
    # print(DataFields[1:8])
    DataTrain=numpy.zeros(numpy.shape(Train_Pred))
    DataInRescale=numpy.zeros(numpy.shape(In_Pred))
    
    DataTrain=(Train_Pred-Train_Pred.mean(axis=0))/ Train_Pred.std(axis=0, ddof=0)
    maxminAll=DataTrain.max(axis=0)-DataTrain.min(axis=0)
    DataTrain*=(maxminAll.max(axis=0)/(maxminAll))
    # DataAllRescale[:,1:8]=KNN_Functions.rescale(DataAll[:,1:8])
    
    DataInRescale=(In_Pred-Train_Pred.mean(axis=0))/ Train_Pred.std(axis=0, ddof=0)
    DataInRescale*=(maxminAll.max(axis=0)/(maxminAll))
    # DataInRescale[:,1:8]=KNN_Functions.rescalebis(DataAll[:,1:8],DataIn[:,1:8])
    
    ## Calculate distances

    # for testing purposes
    if Knumber == -1 :
        Knumber=int(numpy.round(0.7244*(nTrain)**0.468))
    else:
        Knumber = int(Knumber)

    if Power == -1 : 
        Power=(numpy.round((0.7669*(nTrain)**0.049)*100))/100
    
    Distance=numpy.empty((nTrain,nPar,nIn)) #7:"Clay", "Silt", "Sand", "BD", "OC", "pH", "CEC"
    
    for j in range(0,nIn):
        Distance[:,:,j]=(DataTrain-DataInRescale[j,:])**2 # DISTANCE**2 traindata x soil parameter x input
    
    DistanceTotal=numpy.sum(Distance,axis=1) 
    DistanceTotal=numpy.sqrt(numpy.abs(DistanceTotal)) # DISTANCE traindata x input
    
    if LeaveOneOut:
        DistanceTotal[DistanceTotal==0]=1e9 # Make distance high, so identical train-input samples are left out
    
    KNearestNeighbourIndex = DistanceTotal.argsort(axis=0)[0:Knumber,:] # INDEX Nearest x Input
    
    KNearestNeighbourDistance=numpy.empty(KNearestNeighbourIndex.shape)# DISTANCE Nearest x Input
    KNearestNeighbour_Est = numpy.empty((numpy.hstack((KNearestNeighbourIndex.shape,numpy.size(Train_Est, axis=1)))))# SWRC Nearest x Input x pF

    for x in range(0,nIn):
        KNearestNeighbourDistance[:,x]=DistanceTotal[KNearestNeighbourIndex[:,x],x]
        KNearestNeighbour_Est[:,x,:]=Train_Est[KNearestNeighbourIndex[:,x],:] 
    
    KNearestNeighbourDistance[KNearestNeighbourDistance==0]=10e-8 # replace zeros to avoid infinity
    
    ## Assign Weights
    
    DistanceSum=KNearestNeighbourDistance.sum(axis=0)
    DistanceWeight=(DistanceSum/KNearestNeighbourDistance)**Power
    DistanceWeightScale=DistanceWeight/(DistanceWeight.sum(axis=0)) #WEIGHTS Nearest x Input
    DistanceWeightScale=numpy.expand_dims(DistanceWeightScale,2)
    DistanceWeightScale=numpy.repeat(DistanceWeightScale, nEst, axis=2) #WEIGHTS Nearest x Input x  pF
    
    ## Calculate Estimates
    
    KNNEstimate=numpy.sum(KNearestNeighbour_Est*DistanceWeightScale,axis=0)

    return KNNEstimate


