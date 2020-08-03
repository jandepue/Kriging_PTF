#==============================================================================
# Demonstration of Kriging and kNN pedotransfer function
# Created By Jan De Pue (2020) at Ghent University
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
# from Krig_Funky import *
from Validation_Funk import *
from GPI_Funk import *

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels

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
    RandomSample = numpy.array(random.sample(range(nS),nTrainSeparation))
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


## Normalize
TrainData_In_mean = numpy.mean(TrainData_In_0,axis=0)
TrainData_In_std = numpy.std(TrainData_In_0,axis=0)

TrainData_In = (TrainData_In_0 - TrainData_In_mean)/(TrainData_In_std)
InputData_In = (InputData_In_0 - TrainData_In_mean)/(TrainData_In_std)
TrainData_Out = TrainData_Out_0
InputData_Out = InputData_Out_0

#==============================================================================
# GPL
#==============================================================================

OptimizedKernel_L = []
OptimizedRegression_L = []
theta_L = []

print('Fit Kernel')
for iOut in range(nOut):
    # Kernel selection: Anisotropic
    var = numpy.var(TrainData_Out[:,iOut])
    kernel = kernels.ConstantKernel(var/2, constant_value_bounds=(var*1e-3, var*1e1)) \
                * kernels.RBF(length_scale=[1.0,]*nIn, length_scale_bounds=(1e-2, 1e3)) \
                + kernels.WhiteKernel(noise_level=var/2, noise_level_bounds=(var*1e-3, var*1e0))
    ## Fit
    nFit = 15
    gp = GaussianProcessRegressor(kernel=kernel,
                                    n_restarts_optimizer=nFit,
                                    normalize_y = True,
                                    alpha=0.0).fit(TrainData_In,TrainData_Out[:,iOut])
    OptimizedKernel_L.append(gp.kernel_)
    OptimizedRegression_L.append(gp)
    theta_L.append(gp.kernel_.theta)
    theta_prev = gp.kernel_.theta

    ## Plot kernel
    nl = 100
    lagplot = numpy.logspace(-1,3,nl)
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    for iIn in range(nIn):
        Y = numpy.zeros((nl,nIn))
        Y[:,iIn] = lagplot
        cov = Kernel_RBF_WN(gp.kernel_.theta,Y)
        ax.plot(lagplot,cov[0,:],label=Header_In[iIn])
    ax.set_xscale('log')
    ax.set_xlabel('lag (-)')
    ax.set_ylabel('covariance (-)')
    ax.legend()
    figlist.append(fig)

## Plot length scales
lscales = numpy.exp(numpy.array(theta_L)[:,1:nIn+1])
figlist.append(fig)
fig = pylab.figure()
ax = fig.add_subplot(111)
for iIn in range(nIn):
    ax.plot(range(nOut),lscales[:,iIn],label=Header_In[iIn])
ax.set_yscale('log')
ax.set_xticks(range(nOut))
ax.set_xticklabels(Header_Out)
ax.set_xlabel('Length scale (-)')
ax.set_ylabel('Response Variable')
ax.legend()
figlist.append(fig)

print('Kernel fit: Done')

#==============================================================================
# Ordinary Kriging
#==============================================================================
print('Prediction')

k_All = 50

## BOOTSTRAPPING
nB = 10
BootSize=0.8
# nB = 1
# BootSize=1.0

GPIEst_Boot=numpy.empty((nInput,nOut,nB))
GPIStd_Boot=numpy.empty((nInput,nOut,nB))
KNNEst_Boot=numpy.empty((nInput,nOut,nB))

for iB in range(nB):

    print("Bootstrap %s / %s"%(iB,nB-1))
    nSamp=int(nTrain*BootSize)# part of dataset used for resampling to measure statistics
    RandUnique=random.sample(range(nTrain), nSamp) # Generate unique random numbers
    TrainData_In_Samp=TrainData_In[RandUnique,:]
    TrainData_Out_Samp=TrainData_Out[RandUnique,:]
    TrainData_In_0_Samp=TrainData_In_0[RandUnique,:]
    TrainData_Out_0_Samp=TrainData_Out_0[RandUnique,:]

    ## GPL
    CovKernel = Kernel_RBF_WN
    GPIEst_Boot[:,:,iB], GPIStd_Boot[:,:,iB] = GPI(TrainData_In_Samp,
                                                   TrainData_Out_Samp,
                                                   InputData_In,
                                                   theta_L,
                                                   CovKernel,
                                                   nNeighbours_L = (numpy.ones(nOut)*k_All).astype(int),
                                                   LeaveOneOut = LeaveOneOut)

    ## KNN
    Knn_Power_used = numpy.zeros(nOut)
    for iO in range(nOut):
        Knn_k = k_All
        Knn_Power =  2.0 ## => this value should be optimized

        ## Optimize k power
        # Knn_Power_init =  2.0
        # Knn_Min = minimize(kNN_Fit,
        #     x0 = (Knn_Power_init,),
        #     args = (TrainData_In_0_Samp,
        #           InputData_In_0,
        #           TrainData_Out_0_Samp[:,iO][:,None],
        #           InputData_Out_0[:,iO][:,None],
        #           Knn_k),
        #     method = 'SLSQP',
        #     bounds = ((0,6),),
        #     options = {'maxiter' : 20, 'disp':True},
        #     )
        # Knn_Power = Knn_Min.x[0]
        # print('%s => %s'%(Knn_Power_init,Knn_Power))

        KNNEst_Boot[:,iO,iB] = KNN(TrainData_In_0_Samp,
                                  InputData_In_0,
                                  TrainData_Out_0_Samp[:,iO][:,None],
                                  LeaveOneOut = LeaveOneOut,
                                  Knumber = Knn_k,
                                  Power = Knn_Power)[:,0]

## postprocess
GPI_Est = GPIEst_Boot.mean(axis=2)
GPI_Std = GPIStd_Boot.mean(axis=2)
KNN_Est = KNNEst_Boot.mean(axis=2)

GPI_Est_BootSTD = GPIEst_Boot.std(axis=2).mean(axis=0)
GPI_Std_BootSTD = GPIStd_Boot.std(axis=2).mean(axis=0)
KNN_Est_BootSTD = KNNEst_Boot.std(axis=2).mean(axis=0)

GPI_Est_BootSTD_All = GPIEst_Boot.std(axis=2)
KNN_Est_BootSTD_All = KNNEst_Boot.std(axis=2)

#==============================================================================
# Validation
#==============================================================================

GPI_Error = GPI_Est - InputData_Out_0
KNN_Error = KNN_Est - InputData_Out_0

GPI_RelError = GPI_Error/InputData_Out_0
KNN_RelError = KNN_Error/InputData_Out_0

GPI_ME = numpy.mean(GPI_Error,axis=0)
KNN_ME = numpy.mean(KNN_Error,axis=0)

GPI_RMSE = numpy.sqrt(numpy.sum(GPI_Error**2,axis=0)/nInput)
KNN_RMSE = numpy.sqrt(numpy.sum(KNN_Error**2,axis=0)/nInput)

GPI_RelRMSE = numpy.sqrt(numpy.sum(GPI_RelError**2,axis=0)/nInput)
KNN_RelRMSE = numpy.sqrt(numpy.sum(KNN_RelError**2,axis=0)/nInput)

GPI_R2 = PearsonR2(InputData_Out_0,GPI_Est,axis = 0)
KNN_R2 = PearsonR2(InputData_Out_0,KNN_Est,axis = 0)

GPI_NS = NashSutcliffeMEC(InputData_Out_0,GPI_Est,axis = 0)
KNN_NS = NashSutcliffeMEC(InputData_Out_0,KNN_Est,axis = 0)

#==============================================================================
# Plot
#==============================================================================

fig = pylab.figure(figsize = [14,6])
ax = fig.add_subplot(121)
for iO in range(nOut):
    color=cmap(iO/(nOut-0.99))
    ax.errorbar(InputData_Out_0[:,iO], KNN_Est[:,iO],
                yerr = KNNEst_Boot.std(axis=2)[:,iO],
                fmt='.',color = color, label = Header_Out[iO])
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
    ax.errorbar(InputData_Out_0[:,iO], GPI_Est[:,iO],
                yerr = GPIEst_Boot.std(axis=2)[:,iO],
                fmt='.',color = color, label = Header_Out[iO])
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


width=0.4
fig = pylab.figure()
ax = fig.add_subplot(111)
ax.bar(numpy.arange(nOut)-width,KNN_NS,width=width,align='edge',label='kNN')
ax.bar(numpy.arange(nOut),GPI_NS,width=width,align='edge',label='Kriging')
ax.set_ylim([0,1])
ax.set_xticks(numpy.arange(nOut))
ax.set_xticklabels(Header_Out)
ax.legend(loc=4)
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

