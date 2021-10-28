import numpy as np
import numba as nb
import pdb

@nb.njit
def unfold_trajectories_njit(pos, box_size):
    """
    Inputs:
    -------------
    pos: 3D float array
        Size is nDim x nObjects x nTime
    box_size: list or 1D array
        size is nDim x 1
    """

    pos_unfolded = np.zeros_like(pos)

    # Loop over objects
    for jobj in np.arange(pos.shape[1]):

        # For each object, unfold it's coordinates
        pos_unfolded[:,jobj,:] = unfold_trajectory_njit( pos[:,jobj,:], box_size)
    
    return pos_unfolded

@nb.njit
def unfold_trajectory_njit(crds, box_size):
    """
    Inputs:
    -------------
    pos: 3D float array
        Size is nDim x x nTime
    box_size: list or 1D array
        size is nDim x 1
    """
    # unfolded crds via the nearest image convention

    # Note that we transpose so that dimensional order is nTime x nDim
    crds = crds.transpose()

    # reference coordinate
    crds_unfolded = np.zeros_like(crds)
    crds_unfolded[0,:] = crds[0,:]
    for jdim in np.arange(crds.shape[1]):
        L = box_size[jdim]
        for jt in np.arange(1,crds.shape[0]):
            Xi = (crds_unfolded[jt-1,jdim] - crds[jt,jdim]) + L/2.0
            Xi = Xi - L*np.floor(Xi/L)
            d = (L/2.0) - Xi
            crds_unfolded[jt,jdim] = crds_unfolded[jt-1,jdim] + d
    return crds_unfolded.transpose()

