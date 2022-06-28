#!/usr/bin/env python
# coding: utf-8

# # A function to match the DM delays of the FDMT

# In[1]:


from pylab import *
import matplotlib as mpl
import numpy as np
from scipy import constants
sys.path.append('../craft_craco/src/craft/')

import sys
sys.path.append("/home/gup037/Codes/craft/src/craft/")

import fdmt # you'll need to have ../python in  you PYTHONPATH
from collections import OrderedDict

def get_dm_mask(f_lower=0.976,chanbwGhz=1e-3,Nchan=256,NDMtrials=1024,
        TimeBlockSize=256,wanted_idm=900, plot=False, doList=False):
    """
    Generates an array giving the cells which are summed to search for an FRB.
    
    
    Inputs:
        f_lower: lower edge of frequency band (in GHz). Calculated from central
            frequency, minus half the total bandwidth. Here, 1.12-288/2=0.976
        chanbwGHz: channel bandwidth in GHz. Frequencies are increments from f1
            upwards in units of chanbwGHz
        Nchan: number of frequency channels
        NDMtrials: number of DM trials, from 0 to NDMtrials-1 time increments
        TimeBlockSize: time length of each block, in units of integration time
        wanted_idm: number of time increments over which a pulse is dispersed between
            lowest and highest channel for which the mask is generated.
        plot (bool): generate a plot of the mask
        doList (bool): if True return a list of indicaes for each channel
    
    Returns:
        mydmmask: 2D Numpy array. Dimensions are Nchan x ~dm_mask length (not determined).
            Cells containing 1 indicate that the cell is being summed.
            Cells containing 0 indicate the cell is ignored.
            This mask is later convolved with a width search which we must
                correct for.
        thelist: list of included offsets for each channel
    """
    
    thefdmt=init_fdmt(f_lower,chanbwGhz,Nchan,NDMtrials,TimeBlockSize)
    myfrbmask = thefdmt.add_frb_track(wanted_idm)
    #print("shape is ",myfrbmask.shape)
    if plot:
        imshow(myfrbmask, aspect='auto', origin='lower')
        plt.xlabel('time')
        plt.ylabel('frequency')
        #plt.savefig('plot_of_frb_mask.pdf')
        plt.show()
        plt.close()
    if doList:
        # make a list of things
        thelist=[]
        for ichan in np.arange(Nchan):
            OK=np.where(myfrbmask[ichan]==1.0)[0]
            thelist.append(OK)
        return thelist
    else:
        return myfrbmask
    
def init_fdmt(f_lower=0.976,chanbwGhz=1e-3,Nchan=256,NDMtrials=1024, TimeBlockSize=256):
    """
    Function to initialise an FDMT object
    Currently hard-codes inputs
    
    Inputs:
        f_lower: lower edge of frequency band (in GHz). Calculated from central
            frequency, minus half the total bandwidth. Here, 1.12-288/2=0.976
        chanbwGHz: channel bandwidth in GHz. Frequencies are increments from f1
            upwards in units of chanbwGHz
        Nchan: number of frequency channels
        NDMtrials: number of DM trials, from 0 to NDMtrials-1 time increments
        TimeBlockSize: time length of each block, in units of integration time
    
    """
    # original code - not used here
    #fc = 1.12 # center frequency GHz
    #bw=1e-3
    #f1 = fc - bw/2.
    #f2 = fc + bw/2.
    #bw = 0.288 # bandwidth GHz
    #Tint = 0.864e-3 # integration time - seconds
    
    import importlib
    importlib.reload(fdmt)
    thefdmt = fdmt.Fdmt(f_lower, chanbwGhz, Nchan, NDMtrials, TimeBlockSize)
    return thefdmt



# calls the routine
if __name__ == '__main__':
    main()
