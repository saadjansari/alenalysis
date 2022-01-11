from numba import njit
import numpy as np
import src.decorators
from src.DataHandler import *
import pdb
import scipy.spatial.ckdtree

def PlotLocalPolarOrderVsTime( FData, savepath):
    """ Plot local polar order for each filament vs time """

    print('Plotting Local Polar Order Per Time Image (via KDtree)...')
    PolarOrder = FData.local_polar_order_

    fig,ax = plt.subplots()

    im = ax.imshow(PolarOrder, cmap='viridis', interpolation='nearest', 
            aspect=0.3, vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set(xlabel='Frame', ylabel='Filament', 
            title='Local Polar Order KD')

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

def PlotLocalPolarOrderHistogram( FData, savepath):
    """ Plot local polar order histogram over all of time (KDtree) """

    print('Plotting Local Polar Order Histogram (via KDtree)...')
    PolarOrder = FData.local_polar_order_

    fig,ax = plt.subplots()
    ax.hist(PolarOrder.flatten(), bins=np.linspace(-1,1,50))[-1]
    ax.set(yscale='log', xlabel='Local Polar Order KD', ylabel='Counts' )
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

def PlotLocalNematicOrderVsTime( FData, savepath):
    """ Plot local nematic order for each filament vs time """

    print('Plotting Local Nematic Order Per Time Image (via KDtree)...')
    SOrder = FData.local_nematic_order_

    fig,ax = plt.subplots()

    im = ax.imshow(SOrder, cmap='viridis', interpolation='nearest', 
            aspect=0.3, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set(xlabel='Frame', ylabel='Filament', 
            title='Local Nematic Order')

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

def PlotLocalNematicOrderHistogram( FData, savepath):
    """ Plot local nematic order histogram over all of time (KDtree) """

    print('Plotting Local Nematic Order Histogram (via KDtree)...')
    SOrder = FData.local_nematic_order_

    fig,ax = plt.subplots()
    ax.hist(SOrder.flatten(), bins=np.linspace(0,1,50))[-1]
    ax.set(yscale='log', xlabel='Local Nematic Order', ylabel='Counts' )
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

def calc_local_order( c, orients, boxsize, max_dist):
    """
    Calculate the local polar order via KDtree
    Inputs: 
        c: coorindates 3 x N x T (where N is number of filaments, T is number of frames)
        orient_array : 3 x N x T 
        boxsize : 1 x 3
        max_dist: float (max distance threshold defining local region)
    """
    # Number of frames 
    n_frames = c.shape[2]

    # local polar order array
    Porder = np.zeros( (c.shape[1], c.shape[2]) )
    Sorder = np.zeros( (c.shape[1], c.shape[2]) )

    for jframe in range(0,n_frames):
        print('Frame = {0}/{1}'.format(jframe,n_frames-1) )
        Porder[:,jframe], Sorder[:,jframe] = calc_local_order_frame( c[:,:,jframe], orients[:,:,jframe], boxsize, max_dist)
    return Porder, Sorder

@src.decorators.timer
# @njit(parallel=True)
def calc_local_order_frame( c, orients, boxsize, max_dist):
    """
    Calculate the local polar order using KDTree
    Inputs: 
        c: coorindates 3 x N (where N is number of filaments, T is number of frames)
        orient_array : 3 x N 
        boxsize : 1 x 3
        max_dist: float (max distance threshold defining local region)
    """

    Porder = np.zeros(c.shape[1])
    Sorder = np.zeros(c.shape[1])

    # Q tensor for all filaments
    Q_all = np.diagonal( np.tensordot( orients[:,:], orients[:,:], axes=0), axis1=1, axis2=3)

    # find distance between points
    kdtree = scipy.spatial.cKDTree( c.transpose(), boxsize=boxsize)

    for idx in np.arange( c.shape[1]):

        dists, idxs = kdtree.query( c[:,idx],k=100, distance_upper_bound=max_dist)
        idx_max = np.where( dists==np.inf)[0]
        if len(idx_max) == 0:
            idxs = idxs[1:]
            qt = np.exp( -1*dists[1:]**4)
        else:
            idxs = idxs[1:idx_max[0]]
            qt = np.exp( -1*dists[1:idx_max[0]]**4)

        if np.sum(qt) == 0:
            Porder[idx] = np.nan 
            Sorder[idx] = np.nan 
        else:
        
            # orientation of reference filament
            o1 = orients[:,idx]
            
            # orientation of nearby filaments
            dp = (orients[:,idxs].transpose() *o1).transpose()
            scaled_qq = qt / np.sum( qt)
            mp = scaled_qq * dp
            Porder[idx] = np.sum(mp)

            # Average weighed Q tensor
            Q_local = np.average( Q_all[:,:,idxs], axis=2, weights=qt) - np.identity(3)/3
            Sorder[idx] = np.sqrt(np.tensordot(Q_local, Q_local)*1.5)

    return Porder, Sorder
