#==============================================================================
# Kriging Functions
# Created By Jan De Pue (2018) at Ghent University
# Reference:
#==============================================================================
import numpy
import copy

## VARIOGRAM

def VariogramModel1(x,Par):
    # Spheric

    C0,C1,R = Par

    V = C0 + C1*(1.5*x/R - 0.5*(x/R)**3)
    V[x>=R] = C0+C1
    V[x==0] = 0.0
    return V

def VariogramModel2(x,Par):
    # Exponential
    
    C0,C1,R = Par

    V = C0 + C1 * (1.0- numpy.exp(-x**2 / R**2))
    
    V[x==0] = 0.0
    return V

def VariogramModel2b(x,Par):
    # Exponential
    
    C0,C1,R = Par

    V = C0 + (C1-C0) * (1.0- numpy.exp(-x**2 / R**2))
    
    V[x==0] = 0.0
    return V

def VariogramModel3(x,Par):
    # Polynomal 3th degree
    
    P0,P1,P2,P3 = Par

    V = P0 + P1*x + P2*x**2 + P3*x**3 
    V[x==0] = 0.0
    
    return V

def VariogramModel4(x,Par):
    # Polynomal 1th degree
    
    P0,P1 = Par

    V = P0 + P1*x
    V[x==0] = 0.0
    
    return V

def ObjectiveFunc1(Par,x,Y_True,model=VariogramModel1):
    # For fitting
    
    Y_Model=model(x,Par)
    Diff=(Y_Model - Y_True)**2
    return Diff.sum()

def ObjectiveFunc2(Par,x,Y_True,model=VariogramModel1):
    # For fitting
    
    Y_Model=model(x,Par)
    Diff=((Y_Model - Y_True)/Y_True)**2
    return Diff.sum()

def ObjectiveFunc3(Par,x,Y_True,model=VariogramModel1):
    # For fitting
    
    Y_Model=model(x,Par)
    Diff = Y_Model - Y_True
    return Diff

def ObjectiveFunc4(Par,x,Y_True,model=VariogramModel1):
    # For fitting
    
    Y_Model=model(x,Par)
    Diff = (Y_Model - Y_True)/Y_True
    return Diff

def ExperimentalVariogram(dTrain_In,dTrain_Out,LagRange):
    
    LagRange_M = (LagRange[:-1] + LagRange[1:])/2
    nL = LagRange_M.size
    nIn = dTrain_In.shape[-1]
    nOut = dTrain_Out.shape[-1]
    ## Variogram

    SemiVariogram = numpy.zeros((nIn,nOut,nL))+numpy.nan
    SemiVariogram_std = numpy.zeros((nIn,nOut,nL))+numpy.nan

    for iL in range(nL):
        Lmin,Lmax = LagRange[[iL,iL+1]]
        for iI in range(nIn):
            for iO in range(nOut):
                if numpy.sum((dTrain_In[:,:,iI] >= Lmin) & (dTrain_In[:,:,iI] < Lmax)) > 0:
                    SemiVariogram[iI,iO,iL] = numpy.nanmean((dTrain_Out**2)
                                                            [:,:,iO]
                                                            [(dTrain_In[:,:,iI] >= Lmin) & (dTrain_In[:,:,iI] < Lmax)])/2
                    SemiVariogram_std[iI,iO,iL] = numpy.nanstd((dTrain_Out**2)
                                                               [:,:,iO]
                                                               [(dTrain_In[:,:,iI] >= Lmin) & (dTrain_In[:,:,iI] < Lmax)])/2
                    
                ## AVOID NAN: replace by previous value
                else:
                    SemiVariogram[iI,iO,iL] = SemiVariogram[iI,iO,iL-1]


    ## AVOID NAN: replace by next value
    for iL in range(nL)[::-1]:
        for iI in range(nIn):
            for iO in range(nOut):
                if numpy.isnan(SemiVariogram[iI,iO,iL]):
                    if iL == nL-1:
                        SemiVariogram[iI,iO,iL] = SemiVariogram[iI,iO,~numpy.isnan(SemiVariogram[iI,iO,:])][-1]
                    else:
                        SemiVariogram[iI,iO,iL] = SemiVariogram[iI,iO,iL+1]

    return SemiVariogram, SemiVariogram_std


def FitVariogramModel(SemiVariogram,LagRange,VarModel,Par0_L = -1, bnd_L  = -1, nP=-1):
    from scipy.optimize import minimize,least_squares
    
    LagRange_M = (LagRange[:-1] + LagRange[1:])/2

    nIn,nOut,nLag = SemiVariogram.shape
    
    if nP ==-1:
        nP = len(Par0_L[0][0])

    SemiVariogram_Par = numpy.zeros((nIn,nOut,nP))
    
    for iO in range(nOut):
        for iI in range(nIn):
            if Par0_L == -1:
                Par0 = [SemiVariogram[iI,iO,0],SemiVariogram[iI,iO,-1],1]
            else:
                Par0 = Par0_L[iI][iO]
            if bnd_L == -1:
                bnd=((0,None),(0,None),(0,5))
            else:
                bnd = bnd_L[iI][iO]
                
            Opt = least_squares(ObjectiveFunc3, Par0,
                                args=(LagRange_M,SemiVariogram[iI,iO,:],VarModel),
                                loss= 'linear',
                                # loss = 'soft_l1',
                                f_scale=0.10,
                                max_nfev = 10000,
                                verbose = 0,
                                bounds = bnd
                                )
            ParFit=Opt.x 
        
            SemiVariogram_Par[iI,iO,:]=ParFit
        
            
    return SemiVariogram_Par


def OrdinaryKriging(TrainData_In,TrainData_Out,InputData_In, SemiVariogram_Par, VarModel, nNeighbours = 50, LeaveOneOut = False):
    ## NO DISTANCE WEIGHTS
    
    # import pdb
    
    nTrain,nIn=TrainData_In.shape
    nTrain,nOut=TrainData_Out.shape
    nInput,nIn=InputData_In.shape

    ## Search Nearest Neighbours

    dTrain_In = abs(TrainData_In[:,None,:] - TrainData_In[None,:,:])
    dTrainInput_In = abs(TrainData_In[:,None,:] - InputData_In[None,:,:])
    dTrainInput_In_TotalDistance = numpy.sqrt(numpy.sum(dTrainInput_In**2,axis=2))/nIn
    if LeaveOneOut:
        dTrainInput_In_TotalDistance[dTrainInput_In_TotalDistance==0] = 1e9
    dTrainInput_In_Sort = numpy.argsort(dTrainInput_In_TotalDistance,axis=0)

    Kriging_Est = numpy.zeros((nInput,nOut))
    Kriging_Std = numpy.zeros((nInput,nOut))
    
    for iIn in range(nInput):
    
        NearestNeighbours = dTrainInput_In_Sort[:nNeighbours,iIn]
        
        dTrain_In_Nearest = dTrain_In[NearestNeighbours,:,:][:,NearestNeighbours,:]
        dTrainInput_In_Nearest = dTrainInput_In[NearestNeighbours,iIn,:]
    
        VarA = numpy.zeros((nNeighbours,nNeighbours,nIn,nOut))
        for iI in range(nIn):
            for iO in range(nOut):
                VarA[:,:,iI,iO] = VarModel(dTrain_In_Nearest[:,:,iI],SemiVariogram_Par[iI,iO,:])
        VarA = numpy.sqrt(numpy.sum(VarA**2,axis = 2))/nIn ## WEIGHTING OVER DIFFERENT DIMENSIONS: Here's where the magic happens

        
        A = numpy.ones((nNeighbours+1,nNeighbours+1,nOut))
        A[:nNeighbours,:nNeighbours,:] = VarA
        A[-1,-1,:]=0

        VarB = numpy.zeros((nNeighbours,nIn,nOut))
        for iI in range(nIn):
            for iO in range(nOut):
                VarB[:,iI,iO] = VarModel(dTrainInput_In_Nearest[:,iI],SemiVariogram_Par[iI,iO,:])
        VarB = numpy.sqrt(numpy.sum(VarB**2,axis = 1))/nIn ## WEIGHTING OVER DIFFERENT DIMENSIONS: Here's where the magic happens
        
        B = numpy.ones((nNeighbours+1,1,nOut))
        B[:-1,0,:] = VarB

        l = numpy.ones((nNeighbours+1,1,nOut))
        for iO in range(nOut):
            l[:,:,iO] = numpy.linalg.inv(A[:,:,iO]).dot(B[:,:,iO])

        Weights = l[:-1,0,:]
        Lagrange = l[-1,0,:]

        Kriging_Est[iIn,:] = numpy.sum(Weights * TrainData_Out[NearestNeighbours,:],axis = 0)
        Kriging_Std[iIn,:] = numpy.sum(Weights * VarB, axis=0) + Lagrange

        # pdb.set_trace()
        
    return Kriging_Est, Kriging_Std


def OrdinaryKriging_Optimal(TrainData_In,TrainData_Out,InputData_In, SemiVariogram_Par, VarModel, nNeighbours_L = -1,DistanceWeights = -1,LeaveOneOut = False):
    import pdb
    
    nTrain,nIn=TrainData_In.shape
    nTrain,nOut=TrainData_Out.shape
    nInput,nIn=InputData_In.shape

    ## MaxNeighbours
    if '%s'%nNeighbours_L =='-1' :
        SumInvNugget = numpy.sum(SemiVariogram_Par[:,:,0]/SemiVariogram_Par[:,:,1],axis=0)
        nNeighbours_L = (1 + 32 * numpy.log10(nTrain)/SumInvNugget).astype(int)
        
    ## Weights
    if '%s'%DistanceWeights =='-1' : 
        # based on nugget effect 
        NuggetEffect = numpy.zeros((nIn,nOut))
        for iI in range(nIn):
            for iO in range(nOut):
                NuggetEffect[iI,iO] = VarModel(numpy.array([1e-2,]),SemiVariogram_Par[iI,iO,:]) / VarModel(numpy.array([1.5,]),SemiVariogram_Par[iI,iO,:])
        DistanceWeights = 1/NuggetEffect**2 # ! only use this to alter the search of neighbours, not the actual distance!
        
    ## Search Nearest Neighbours
    
    dTrain_In = abs(TrainData_In[:,None,:] - TrainData_In[None,:,:])
    dTrainInput_In = abs(TrainData_In[:,None,:] - InputData_In[None,:,:])
    # dTrainInput_In_TotalDistance = numpy.sqrt(numpy.sum((dTrainInput_In)**2,axis=2))/nIn  # Unweighted (nTrain x nInput)
    dTrainInput_In_TotalDistance = numpy.sqrt(numpy.sum((dTrainInput_In[:,:,:,None] * DistanceWeights[None,None,:,:])**2,axis=2))/nIn # weighted (nTrain x nInput x nOut)
    if LeaveOneOut:
        dTrainInput_In_TotalDistance[dTrainInput_In_TotalDistance==0] = 1e9
    dTrainInput_In_Sort = numpy.argsort(dTrainInput_In_TotalDistance,axis=0)

    Kriging_Est = numpy.zeros((nInput,nOut))
    Kriging_Std = numpy.zeros((nInput,nOut))
    Kriging_Neighbours = numpy.zeros((nInput,nOut))
    
    for iIn in range(nInput):
        for iO in range(nOut):

            nNeighbours = nNeighbours_L[iO]
            NearestNeighbours = dTrainInput_In_Sort[:nNeighbours,iIn,iO]
                        
            dTrain_In_Nearest = dTrain_In[NearestNeighbours,:,:][:,NearestNeighbours,:]
            dTrainInput_In_Nearest = dTrainInput_In[NearestNeighbours,iIn,:]
            
            VarA = numpy.zeros((nNeighbours,nNeighbours,nIn))
            VarB = numpy.zeros((nNeighbours,nIn))

            for iI in range(nIn):
                VarA[:,:,iI] = VarModel(dTrain_In_Nearest[:,:,iI],SemiVariogram_Par[iI,iO,:])

            for iI in range(nIn):
                VarB[:,iI] = VarModel(dTrainInput_In_Nearest[:,iI],SemiVariogram_Par[iI,iO,:])

            
            VarA = numpy.sqrt(numpy.sum(VarA**2,axis = 2))/nIn ## WEIGHTING OVER DIFFERENT DIMENSIONS
            VarB = numpy.sqrt(numpy.sum(VarB**2,axis = 1))/nIn ## WEIGHTING OVER DIFFERENT DIMENSIONS

            A = numpy.ones((nNeighbours+1,nNeighbours+1))
            A[:nNeighbours,:nNeighbours] = VarA
            A[-1,-1]=0

            B = numpy.ones((nNeighbours+1,1))
            B[:-1,0] = VarB

            NearestTrainData_Out = TrainData_Out[NearestNeighbours,iO]
            
            try:
                l = numpy.linalg.inv(A).dot(B)
            except:
                print('SINGULAR MATRIX')
                print(A)
                print(B)
                print(NearestNeighbours)
                raise NameError('Singular Matrix')
            Weights = l[:-1,0]
            Lagrange = l[-1,0]

            Kriging_Est[iIn,iO] = numpy.sum(Weights * NearestTrainData_Out,axis = 0)
            Kriging_Std[iIn,iO] = numpy.sum(Weights * VarB, axis=0) + Lagrange

    return Kriging_Est, Kriging_Std

    
