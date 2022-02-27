from numba import njit
import numpy as np
import src.decorators
from src.DataHandler import *
# import decorators
# from DataHandler import *
import pdb
import scipy.spatial.ckdtree
import math

# Plotting
def PlotLocalPackingFractionVsTime( FData, savepath):
    """ Plot local packing fraction for each filament vs time """

    print('Plotting Local Packing Fraction Per Time Image...')
    PackFrac= FData.local_packing_fraction_

    fig,ax = plt.subplots()

    im = ax.imshow(PackFrac, cmap='viridis', interpolation='nearest', 
            aspect=0.3)
    plt.colorbar(im, ax=ax)
    ax.set(xlabel='Frame', ylabel='Filament', 
            title='Local Packing Fraction')

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

def PlotLocalPackingFractionHistogram( FData, savepath):
    """ Plot local packing fraction histogram over all of time"""

    print('Plotting Local Packing Fraction Histogram ...')
    PackFrac = FData.local_packing_fraction_

    fig,ax = plt.subplots()
    ax.hist(PackFrac.flatten(), bins=50)[-1]
    ax.set(yscale='log', xlabel='Local Packing Fraction', ylabel='Counts' )
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()
    

def calc_local_packing_fraction( c0, c1, diameter, boxsize):
    """
    Calculate the local packing fraction
    Inputs: 
        c0: coorindates minus-end 3 x N x T (where N is number of filaments, T is number of frames)
        c1: coorindates plus-end 3 x N x T (where N is number of filaments, T is number of frames)
        boxsize : 1 x 3
    """
    # Number of frames 
    n_frames = c0.shape[2]

    # local packing fraction array
    pf = np.zeros( (c0.shape[1], c0.shape[2]) )

    for jframe in range(0,n_frames):
        print('Frame = {0}/{1}'.format(jframe,n_frames-1) )
        pf[:,jframe] = calc_local_packing_fraction_frame( c0[:,:,jframe], 
                c1[:,:,jframe], diameter, boxsize)
    return pf

@src.decorators.timer
# @decorators.timer
# @njit(parallel=True)
def calc_local_packing_fraction_frame( c0, c1, diameter, boxsize, sampling=5):
    """
    Calculate the local packing fraction for single time frame
    Inputs: 
        c0: coorindates minus-end 3 x N (where N is number of filaments )
        c1: coorindates plus-end 3 x N (where N is number of filaments )
        boxsize : 1 x 3
    """
    pf = np.zeros(c0.shape[1])

    # filament lengths
    fil_len = np.linalg.norm(c1-c0, axis=0)

    # sample points between minus and plus ends
    cs = np.linspace(c0, c1, sampling, axis=1).reshape((c0.shape[0], -1), order='F')
    idxLine = np.array([int(np.floor(i/sampling)) for i in range(cs.shape[-1])])

    # ensure points are within boxsize
    for jdim in range(cs.shape[0]):
        cs[jdim, cs[jdim,:]< 0] +=boxsize[jdim]
        cs[jdim, cs[jdim,:]>= boxsize[jdim]] -= boxsize[jdim]
        cs[jdim, cs[jdim,:]< 0] +=boxsize[jdim]
        cs[jdim, cs[jdim,:]>= boxsize[jdim]] -= boxsize[jdim]
    
    # Initialize kdtree
    kdtree = scipy.spatial.cKDTree( cs.transpose(), boxsize=boxsize)

    # find distance to 50 closest points using ckdtree
    dists, idxNeigh = kdtree.query( cs.transpose(),k=sampling*6)
    idxNeigh = idxLine[idxNeigh.flatten()].reshape(idxNeigh.shape)

    # set distances to self particles to nan
    for jrow in range(idxNeigh.shape[1]):
        dists[jrow, idxNeigh[jrow,:]==0] = np.nan

    # Mean over just the points not in reference line segment
    for idx in np.arange( c0.shape[1]):
        idx_query = np.arange(idx*sampling , idx*sampling + sampling)

        # get indices that are not part of current filament
        idxSample = idxNeigh[idx_query,:].flatten()
        d_collision = np.nanmean(dists[idx_query,:])

        vol_occupied = math.pi * fil_len[idx]* (d_collision/2.)**2 + (4/3)*math.pi*(d_collision/2.)**3
        vol_fil = math.pi * fil_len[idx]* (diameter/2.)**2 + (4/3)*math.pi*(diameter/2.)**3
        pf[idx] = vol_fil/vol_occupied

    return pf

