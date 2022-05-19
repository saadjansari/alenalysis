import numpy as np
from src.CalcOrderParameters import calc_nematic_tensor
from src.calc_com_pbc import calc_com_pbc

def calc_bundle_twist(pos, orient, box_size):
    """
    pos: center of mass position of filaments 3 x N
    orient: orienation of filaments 3 x N
    box_size: size of periodic box 3 x 1
    """

    # Find nematic tensor
    Q = calc_nematic_tensor( orient)

    # Find nematic director
    eigval, eigvec = np.linalg.eig(Q)

    # Nematic director is the eignevector correponding to the largest eigenvalue
    idxMax = np.where( eigval == max(eigval) )[0][0]
    director = eigvec[:,idxMax]

    # Switch nematic director so that it has a positive z component
    # if director[-1] < 0:
        # director *= -1

    # Switch filament orientations such that do tproduct with nematic director is positive
    idxSwitch = np.sum(director.reshape(-1,1)*orient, axis=0) < 0
    orient[:,idxSwitch] *= -1.0

    # Find center of mass
    com = calc_com_pbc( pos.T, box_size)
    
    """ For each filament, find:
        1. distance from com
        2. radial vector r_hat
        3. twist parameter eta, defined as:
            eta = dot( orient, cross(director,r_hat) )
    """ 
    # Radial distance / vector
    rdist = pos - com.reshape(-1,1)
    # apply PBC
    for jdim,bs in enumerate(box_size):
        rdist[jdim,:][ rdist[jdim,:] <= -0.5*bs] += bs
        rdist[jdim,:][ rdist[jdim,:] > 0.5*bs] -= bs

    # Radial distance
    dd = np.linalg.norm( rdist, axis=0)
    
    # Radial unit vectors
    r_hat = rdist / dd
    
    # Twist parameter
    eta = np.sum( orient*np.cross(r_hat.T, director).T, axis=0)
    
    return eta, dd
    
