#==============================================================================
# Demonstration of Kriging and kNN pedotransfer function
# Created By Jan De Pue (2018) at Ghent University
# Reference:
#==============================================================================

import pylab
import numpy
import copy
import random
import sys
import os
import inspect

from KNN_Function import *
from Krig_Funky import *
from Validation_Funk import *

from matplotlib.backends.backend_pdf import PdfPages
figlist=[]
figsize1=[10,10]
figsize2=[10,10./numpy.sqrt(2)]
figsize3=[10,50]
#dpi=None
dpi=300

font = {'family' : 'monospace',
        'size'   : 12}
params = {'mathtext.default': 'regular',
        }
pylab.rc('font', **font)
pylab.rcParams.update(params)
pylab.ioff()

cmap = pylab.get_cmap('Paired')

ScriptBasename = os.path.basename(inspect.getfile(inspect.currentframe()))[:-3]
figlist = []

#==============================================================================
# Specify Dataset
#==============================================================================

Label = 'Unsoda'
TrainFilename = 'SoilData/Unsoda_Export.txt'
InputFilename = 'SoilData/Unsoda_Export.txt'
IDCol = [0,]
InCol = [1,2,3,4]
OutCol = [9,10,11,12]
InLog = []
OutLog = []

print(Label)


#==============================================================================
# Open Data
#==============================================================================
print('Open Data')

fID=open(TrainFilename)
DataFields=numpy.array(fID.readline().replace('\r\n','\n').replace('\n','').split('\t'))
fID.close()

Header_ID=DataFields[IDCol]
Header_In=DataFields[InCol]
Header_Out=DataFields[OutCol]

InputData_ID = numpy.genfromtxt(InputFilename,delimiter='\t',usecols=(IDCol),skip_header=1,dtype='str')

## Option 1: Jacknife
nS = InputData_ID.shape[0]
nTrainSeparation = int(nS*0.7)

if TrainFilename==InputFilename:
    RandomSample = numpy.array(random.sample(numpy.arange(nS),nTrainSeparation))
    RandomFilt = numpy.zeros(nS,dtype='bool')
    RandomFilt[RandomSample] = True

## Option 2: Leave One Out
LeaveOneOut = True
# LeaveOneOut = False


# Training data
if (TrainFilename==InputFilename) & (LeaveOneOut == False): # seperate the dataset in two
    TrainData_In_0 = numpy.genfromtxt(TrainFilename,delimiter='\t',usecols=(InCol),skip_header=1)[RandomFilt,:]
    TrainData_Out_0 = numpy.genfromtxt(TrainFilename,delimiter='\t',usecols=(OutCol),skip_header=1)[RandomFilt,:]
else:
    TrainData_In_0 = numpy.genfromtxt(TrainFilename,delimiter='\t',usecols=(InCol),skip_header=1)
    TrainData_Out_0 = numpy.genfromtxt(TrainFilename,delimiter='\t',usecols=(OutCol),skip_header=1)

# Input data
if (TrainFilename==InputFilename) & (LeaveOneOut == False): # seperate the dataset in two
    InputData_In_0 = numpy.genfromtxt(InputFilename,delimiter='\t',usecols=(InCol),skip_header=1)[~RandomFilt,:]
    InputData_Out_0=numpy.genfromtxt(InputFilename,delimiter='\t',usecols=(OutCol),skip_header=1)[~RandomFilt,:]
else:
    InputData_In_0 = numpy.genfromtxt(InputFilename,delimiter='\t',usecols=(InCol),skip_header=1)
    InputData_Out_0=numpy.genfromtxt(InputFilename,delimiter='\t',usecols=(OutCol),skip_header=1)


## Remove NaN
nanValue = -999
TrainData_In_0[TrainData_In_0 == nanValue] = numpy.nan
TrainData_Out_0[TrainData_Out_0 == nanValue] = numpy.nan
InputData_In_0[InputData_In_0 == nanValue] = numpy.nan
InputData_Out_0[InputData_Out_0 == nanValue] = numpy.nan
nanfilt = ~(numpy.any(numpy.isnan(TrainData_In_0),axis=1) | numpy.any(numpy.isnan(TrainData_Out_0),axis=1))
TrainData_In_0  = TrainData_In_0[nanfilt,:]
TrainData_Out_0 = TrainData_Out_0[nanfilt,:]

nanfilt = ~(numpy.any(numpy.isnan(InputData_In_0),axis=1) | numpy.any(numpy.isnan(InputData_Out_0),axis=1))
InputData_In_0  = InputData_In_0[nanfilt,:]
InputData_Out_0 = InputData_Out_0[nanfilt,:]


## Count
nTrain,nIn=TrainData_In_0.shape
nTrain,nOut=TrainData_Out_0.shape
nInput,nIn=InputData_In_0.shape
# nInput,nOut=InputData_Out_0.shape


## Avoid Singular Matrix (! HAS CONSEQUENCES FOR LEAVE ONE OUT PROCEDURE! )
iPert = -1
Randomizer =  numpy.random.rand(nTrain)
TrainData_In_0[:,iPert] = TrainData_In_0[:,iPert] + TrainData_In_0[:,iPert].mean() * 1e-4 * Randomizer
if (TrainFilename==InputFilename) & (LeaveOneOut == True):
    InputData_In_0[:,iPert] = InputData_In_0[:,iPert] + InputData_In_0[:,iPert].mean() * 1e-4 * Randomizer


## Log Transformations
for iIn in InLog:
    TrainData_In_0[:,iIn] = numpy.log10(TrainData_In_0[:,iIn]+1)
    InputData_In_0[:,iIn] = numpy.log10(InputData_In_0[:,iIn]+1)

for iO in OutLog:
    TrainData_Out_0[:,iO] = numpy.log10(TrainData_Out_0[:,iO]+1)
    InputData_Out_0[:,iO] = numpy.log10(InputData_Out_0[:,iO]+1)

## Boxplot TrainingData
fig=pylab.figure()
fig.suptitle('Original')
ax=fig.add_subplot(211)
bp1=ax.boxplot(TrainData_In_0)
ax.set_xticklabels(Header_In)
ax.set_title('Training Data')
ax=fig.add_subplot(212)
bp1=ax.boxplot(InputData_In_0)
ax.set_xticklabels(Header_In)
ax.set_title('Input Data')
figlist.append(fig)

fig=pylab.figure()
fig.suptitle('Original')
ax=fig.add_subplot(211)
bp1=ax.boxplot(TrainData_Out_0)
ax.set_xticklabels(Header_Out)
ax.set_title('Training Data')
ax=fig.add_subplot(212)
bp1=ax.boxplot(InputData_Out_0)
ax.set_xticklabels(Header_Out)
ax.set_title('Input Data')
figlist.append(fig)

#==============================================================================
# Variogram
#==============================================================================
print('Variogram')

## Normalize

TrainData_In_mean = numpy.mean(TrainData_In_0,axis=0)
TrainData_In_std = numpy.std(TrainData_In_0,axis=0)

TrainData_In = (TrainData_In_0 - TrainData_In_mean)/(TrainData_In_std)
InputData_In = (InputData_In_0 - TrainData_In_mean)/(TrainData_In_std)
TrainData_Out = TrainData_Out_0
InputData_Out = InputData_Out_0


## Distances

dTrain_In = TrainData_In[:,None,:] - TrainData_In[None,:,:]
dTrain_Out = TrainData_Out[:,None,:] - TrainData_Out[None,:,:]


## Lag Classes

# LagMax =1.0
# nL = 25
# LagRange = numpy.linspace(0,LagMax,nL+1)

LagRange = numpy.cumsum((0.005+0.002*numpy.arange(30)))*2
LagRange_M = (LagRange[:-1] + LagRange[1:])/2
nL = LagRange_M.size


## Experimental Variogram

SemiVariogram, SemiVariogram_std = ExperimentalVariogram(dTrain_In,dTrain_Out,LagRange)
MaxSemiVariance =  numpy.mean(numpy.mean(dTrain_Out**2,axis=0),axis=0)/2


## Fit Model

VarModel = VariogramModel2b
Par0_L = []
bnd_L = []

for iI in range(nIn): # Initial parameters and boundaries
    Par0_L_T = []
    bnd_L_T = []
    for iO in range(nOut):
        Par0_L_T.append([min(SemiVariogram[iI,iO,0],MaxSemiVariance[iO]),MaxSemiVariance[iO],0.5])
        bnd_L_T.append([(0,0,1e-2,),(MaxSemiVariance[iO]*1.1,MaxSemiVariance[iO]*1.1,100),])
    Par0_L.append(Par0_L_T)
    bnd_L.append(bnd_L_T)


SemiVariogram_Par = FitVariogramModel(SemiVariogram,LagRange,VarModel,Par0_L = Par0_L, bnd_L  = bnd_L)

# Replace where negative nugget effect occurs!
SemiVariogram_Par[(SemiVariogram_Par[:,:,1] - SemiVariogram_Par[:,:,0])<0,1] = SemiVariogram_Par[(SemiVariogram_Par[:,:,1] - SemiVariogram_Par[:,:,0])<0,0]



## Plot Fit

X = numpy.linspace(0,1.5,100)
for iO in range(nOut):
    fig = pylab.figure()
    ax = fig.add_subplot(111)

    for iI in range(nIn):
        ParFit = SemiVariogram_Par[iI,iO,:]
        Var = VarModel(X,ParFit)
        color = cmap(iI/(nIn-0.999))

        ax.plot(LagRange_M,SemiVariogram[iI,iO,:],'o',color = color,alpha = 0.2)
        ax.plot(X,Var,'-',color = color, label = Header_In[iI])

    ax.set_title('Semivariogram %s'%Header_Out[iO])
    ax.set_ylabel('Semivariogram (-)')
    ax.set_xlabel('Lag (-)')
    ax.legend(loc = 4)
    figlist.append(fig)

NuggetEffect = numpy.zeros((nIn,nOut))
VarioRange = numpy.zeros((nIn,nOut))
RefLag = 1.5
VarioLagTest = numpy.linspace(0,RefLag,1000)
for iI in range(nIn):
    for iO in range(nOut):
        NuggetEffect[iI,iO] = VarModel(numpy.array([1e-2,]),SemiVariogram_Par[iI,iO,:])\
                              / VarModel(numpy.array([RefLag,]),SemiVariogram_Par[iI,iO,:])
        VarioValTest = VarModel(VarioLagTest,SemiVariogram_Par[iI,iO,:])
        VarioRange[iI,iO] = VarioLagTest[VarioValTest >=0.9 * VarModel(numpy.array([RefLag,]),SemiVariogram_Par[iI,iO,:])][0]

width = 1.0/(nIn+1)
fig=pylab.figure(figsize = [15,8])
ax = fig.add_subplot(111)
for iIn in range(nIn):
    color = cmap(iIn/(nIn-0.99))
    xbar = numpy.arange(nOut) + width*iIn
    ax.bar(xbar,NuggetEffect[iIn,:],width*0.9,color=color,label = Header_In[iIn])
ax.set_xlabel('Output')
ax.set_ylabel('Nugget Effect')
ax.set_xticks(range(nOut))
ax.set_xticklabels(Header_Out)
ax.legend()
figlist.append(fig)
    
#==============================================================================
# Ordinary Kriging
#==============================================================================
print('Ordinary Kriging')

## !!!! METAPARAMETER SETTINGS !!!!
Knn_k = (numpy.ones(nOut)*10).astype(int)
Knn_p = numpy.ones(nOut)*2.0

Krig_k = (numpy.ones(nOut)*10).astype(int)
Krig_q = numpy.ones(nOut)*1.0

Krig_DistanceWeights =  1/(NuggetEffect**Krig_q)


## BOOTSTRAPPING
nB = 10
BootSize=0.8
# nB = 1
# BootSize=1.0

Kriging2Est_Boot=numpy.empty((nInput,nOut,nB))
Kriging2Std_Boot=numpy.empty((nInput,nOut,nB))
KNNEst_Boot=numpy.empty((nInput,nOut,nB))

for iB in range(nB):

    print("%s / %s"%(iB,nB-1))
    nSamp=int(nTrain*BootSize)# part of dataset used for resampling to measure statistics (0.935 or 0.8)
    RandUnique=random.sample(range(nTrain), nSamp) # Generate unique random numbers
    TrainData_In_Samp=TrainData_In[RandUnique,:]
    TrainData_Out_Samp=TrainData_Out[RandUnique,:]
    TrainData_In_0_Samp=TrainData_In_0[RandUnique,:]
    TrainData_Out_0_Samp=TrainData_Out_0[RandUnique,:]

    Kriging2Est_Boot[:,:,iB], Kriging2Std_Boot[:,:,iB] = OrdinaryKriging_Optimal(TrainData_In_Samp,
                                                                                 TrainData_Out_Samp,
                                                                                 InputData_In,
                                                                                 SemiVariogram_Par,
                                                                                 VarModel,
                                                                                 nNeighbours_L = Krig_k,
                                                                                 DistanceWeights = Krig_DistanceWeights,
                                                                                 LeaveOneOut = LeaveOneOut)

    for iO in range(nOut):
        KNNEst_Boot[:,iO,iB] = KNN(TrainData_In_0_Samp,
                                   InputData_In_0,
                                   TrainData_Out_0_Samp[:,iO][:,None],
                                   LeaveOneOut = LeaveOneOut,
                                   Knumber = Knn_k[iO],
                                   Power = Knn_p[iO])[:,0]

Kriging2_Est = Kriging2Est_Boot.mean(axis=2)
Kriging2_Std = Kriging2Std_Boot.mean(axis=2)
KNN_Est = KNNEst_Boot.mean(axis=2)

Kriging2_Est_BootSTD = Kriging2Est_Boot.std(axis=2).mean(axis=0)
Kriging2_Std_BootSTD = Kriging2Std_Boot.std(axis=2).mean(axis=0)
KNN_Est_BootSTD = KNNEst_Boot.std(axis=2).mean(axis=0)

Kriging2_Est_BootSTD_All = Kriging2Est_Boot.std(axis=2)
KNN_Est_BootSTD_All = KNNEst_Boot.std(axis=2)

#==============================================================================
# Validation
#==============================================================================

Kriging2_Error = Kriging2_Est - InputData_Out_0
KNN_Error = KNN_Est - InputData_Out_0

Kriging2_RelError = Kriging2_Error/InputData_Out_0
KNN_RelError = KNN_Error/InputData_Out_0

Kriging2_ME = numpy.mean(Kriging2_Error,axis=0)
KNN_ME = numpy.mean(KNN_Error,axis=0)

Kriging2_RMSE = numpy.sqrt(numpy.sum(Kriging2_Error**2,axis=0)/nInput)
KNN_RMSE = numpy.sqrt(numpy.sum(KNN_Error**2,axis=0)/nInput)

Kriging2_RelRMSE = numpy.sqrt(numpy.sum(Kriging2_RelError**2,axis=0)/nInput)
KNN_RelRMSE = numpy.sqrt(numpy.sum(KNN_RelError**2,axis=0)/nInput)

Kriging2_R2 = PearsonR2(InputData_Out_0,Kriging2_Est,axis = 0)
KNN_R2 = PearsonR2(InputData_Out_0,KNN_Est,axis = 0)

Kriging2_NS = NashSutcliffeMEC(InputData_Out_0,Kriging2_Est,axis = 0)
KNN_NS = NashSutcliffeMEC(InputData_Out_0,KNN_Est,axis = 0)



#==============================================================================
# Plot
#==============================================================================

fig = pylab.figure(figsize = [14,6])
ax = fig.add_subplot(121)
for iO in range(nOut):
    color=cmap(iO/(nOut-0.99))
    ax.plot(InputData_Out_0[:,iO], KNN_Est[:,iO],
            '.',color = color, label = Header_Out[iO])

xmin = min(ax.get_xlim()[0],ax.get_ylim()[0])
xmax = max(ax.get_xlim()[1],ax.get_ylim()[1])
xmin = xmin - abs(xmin)*0.1
xmax = xmax + abs(xmax)*0.1
ax.plot([xmin,xmax],[xmin,xmax],'-r')
ax.set_xlim([xmin,xmax])
ax.set_ylim([xmin,xmax])
ax.set_aspect('equal')
ax.set_xlabel('True WC (m3/m3)')
ax.set_ylabel('Predicted WC(m3/m3)')
ax.set_title('kNN')

ax = fig.add_subplot(122)
for iO in range(nOut):
    color=cmap(iO/(nOut-0.99))
    ax.plot(InputData_Out_0[:,iO], Kriging2_Est[:,iO],
            '.',color = color, label = Header_Out[iO])

xmin = min(ax.get_xlim()[0],ax.get_ylim()[0])
xmax = max(ax.get_xlim()[1],ax.get_ylim()[1])
xmin = xmin - abs(xmin)*0.1
xmax = xmax + abs(xmax)*0.1
ax.plot([xmin,xmax],[xmin,xmax],'-r')
ax.set_xlim([xmin,xmax])
ax.set_ylim([xmin,xmax])
ax.set_aspect('equal')
ax.set_xlabel('True WC (m3/m3)')
ax.set_ylabel('Predicted WC(m3/m3)')
ax.set_title('Kriging')

ax.legend(loc = 4)
figlist.append(fig)

#==============================================================================
# Write
#==============================================================================

print('Writing data'.center(50,'='))

import sys
basename=ScriptBasename
postfix = ''

# save plots to pdf
pdfname = '%s%s.pdf'%(basename,postfix)
pp = PdfPages(pdfname)
for fig in figlist:
    pp.savefig(fig)
pp.close()

pylab.show()

