## Gaussian Process Interpolation - i.e. Kriging

import numpy
import copy

## Tools


## Covariance

def Covariance(X,Y):
    cov = numpy.nanmean((X-numpy.nanmean(X))*(Y-numpy.nanmean(Y)))
    return cov

## Fit Kernel (Max Marginal Likelihood)

def LogMarginalLikelihood(theta,Kernel,X,Y):
    # Y = nTrain x 1
    if len(Y.shape)<2:
        Y = Y[:,None]
    # X = nTrain x nPredictors
    d = X.shape[1]

    LML = -0.5* numpy.dot(numpy.dot((Y-numpy.mean(Y)).T, Kernel(theta,X,X)), (Y-numpy.mean(Y)) ) \
        - 0.5*numpy.log(numpy.linalg.det(Kernel(theta,X,X))) \
        - d/2*numpy.log(2*numpy.pi)
    
    return LML

## Kernels

def Kernel_RBF_WN(theta,Y,X = 0):

    nTrain,nIn=Y.shape
    if str(X) == '0':
        X = numpy.zeros((1,nIn))
    nInput,nIn=X.shape

    theta = numpy.exp(theta)
    nu = numpy.sqrt(theta[0])
    l_scale = theta[1:1+nIn]
    wnoise = theta[-1]

    cov = nu**2 * numpy.exp(-0.5*numpy.sum(((Y[None,:,:]-X[:,None,:])/l_scale[None,None,:])**2,axis=2)) + numpy.all(Y[None,:,:]==X[:,None,:],axis=2)*wnoise # nTrain x nInput

    return cov

## GPI

def GPI(TrainData_In,TrainData_Out,InputData_In, theta_L, CovKernel, nNeighbours_L = -1,LeaveOneOut = False):
    
    nTrain,nIn=TrainData_In.shape
    nTrain,nOut=TrainData_Out.shape
    nInput,nIn=InputData_In.shape

    ## MaxNeighbours
    if '%s'%nNeighbours_L =='-1' :
        nNeighbours_L = (numpy.ones(nOut)*20).astype(int)
        # nNeighbours_L = (numpy.ones(nOut)*nTrain/4).astype(int)
        
    ## Search Nearest Neighbours
    TrainVar = numpy.var(TrainData_Out,axis=0)
    dTrain_In = abs(TrainData_In[:,None,:] - TrainData_In[None,:,:])
    dTrainInput_In = abs(TrainData_In[:,None,:] - InputData_In[None,:,:])

    CovDist = numpy.zeros((nTrain,nInput,nOut))
    for iO in range(nOut):
        CovDist[:,:,iO] = -CovKernel(theta_L[iO],InputData_In,TrainData_In)

    if LeaveOneOut:
        CovDist[dTrainInput_In.sum(axis=2)==0] = 1e9
    dTrainInput_In_Sort = numpy.argsort(CovDist,axis=0) ## select samples with largest covariance


    GPI_Est = numpy.zeros((nInput,nOut))
    GPI_Std = numpy.zeros((nInput,nOut))
    GPI_Neighbours = numpy.zeros((nInput,nOut))
    
    for iIn in range(nInput):        
        for iO in range(nOut):
            nNeighbours = nNeighbours_L[iO]
            NearestNeighbours = dTrainInput_In_Sort[:nNeighbours,iIn,iO]
                        
            VarA = numpy.zeros((nNeighbours,nNeighbours,nIn))
            VarB = numpy.zeros((nNeighbours,nIn))

            VarA = CovKernel(theta_L[iO],TrainData_In[NearestNeighbours,:],TrainData_In[NearestNeighbours,:])
            VarB = CovKernel(theta_L[iO],TrainData_In[NearestNeighbours,:],InputData_In[iIn,:][None,:])

            A = numpy.ones((nNeighbours+1,nNeighbours+1))
            A[:nNeighbours,:nNeighbours] = VarA
            A[-1,-1]=0

            B = numpy.ones((nNeighbours+1,1))
            B[:-1,0] = VarB

            NearestTrainData_Out = TrainData_Out[NearestNeighbours,iO]
            
            try:
                l = numpy.linalg.inv(A).dot(B)
            except:
                raise NameError('Singular Matrix')
            Weights = l[:-1,0]
            Lagrange = l[-1,0]

            GPI_Est[iIn,iO] = numpy.sum(Weights * NearestTrainData_Out,axis = 0)
            GPI_Std[iIn,iO] = TrainVar[iO] - l.T.dot(B)
            
    return GPI_Est, GPI_Std



