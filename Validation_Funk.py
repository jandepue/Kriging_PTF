#==============================================================================
# Validation Tools
# Created By Jan De Pue (2018) at Ghent University
# Reference:
#==============================================================================
import numpy

def RMSE(x,y,axis = -1):
    if axis == -1:
        RMSE = numpy.sqrt(numpy.mean((x - y)**2))
    else:
        x = numpy.swapaxes(x,axis,-1)
        y = numpy.swapaxes(y,axis,-1)
        RMSE = numpy.sqrt(numpy.mean((x - y)**2,axis=-1))
    return RMSE
    
def RelRMSE(x,y,axis = -1):
    # x = true
    # y = model
    if axis == -1:
        RMSE = numpy.sqrt(numpy.mean(((x - y)/x)**2))
    else:
        x = numpy.swapaxes(x,axis,-1)
        y = numpy.swapaxes(y,axis,-1)
        RMSE = numpy.sqrt(numpy.mean(((x - y)/x)**2,axis=-1))
    return RMSE
    
def ME(x,y,axis = -1):
    if axis == -1:
        ME = numpy.mean(x - y)
    else:
        x = numpy.swapaxes(x,axis,-1)
        y = numpy.swapaxes(y,axis,-1)
        ME = numpy.mean(x - y,axis = -1)
    return ME
    

def PearsonR2(x,y,axis = -1):
    if axis == -1:
        xmean = numpy.mean(x)
        ymean = numpy.mean(y)
        r2 = numpy.sum((x-xmean)*(y-ymean))/(numpy.sqrt(numpy.sum((x-xmean)**2)) * numpy.sqrt(numpy.sum((y-ymean)**2)))
    else:
        xmean = numpy.expand_dims(numpy.mean(x,axis = axis),axis)
        ymean = numpy.expand_dims(numpy.mean(y,axis = axis),axis)
        r2 = numpy.sum((x-xmean)*(y-ymean),axis = axis)/(numpy.sqrt(numpy.sum((x-xmean)**2,axis = axis)) * numpy.sqrt(numpy.sum((y-ymean)**2,axis = axis)))
    
    return r2

def NashSutcliffeMEC(Ytrue,Ymodel,axis=-1):
    if axis == -1:
        MEC = 1.0 - numpy.sum((Ymodel - Ytrue)**2)/numpy.sum((Ytrue - numpy.mean(Ytrue))**2)
    else:
        Ytrue = numpy.swapaxes(Ytrue,axis,-1)
        Ymodel = numpy.swapaxes(Ymodel,axis,-1)
        MEC = 1.0 - numpy.sum((Ymodel - Ytrue)**2,axis=-1)/numpy.sum((Ytrue - numpy.mean(Ytrue,axis=-1)[...,None])**2,axis=-1)
    return MEC

