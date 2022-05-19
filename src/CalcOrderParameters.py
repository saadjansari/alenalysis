from numba import njit
import numpy as np
from src.DataHandler import *

# Plotting
def PlotNematicOrder(FData, params, savepath):
    """ Plot the global nematic order over time """

    print('Plotting Nematic Order')
    # Nematic order S
    S = calc_nematic_order( FData.orientation_)

    # Display
    timeStep = FData.config_['time_snap']
    timeArray = timeStep * np.arange(FData.nframe_)

    fig,ax = plt.subplots()
    ax.plot(timeArray, S, 'ko', ms=2);
    ax.set_xlim(left=-0.01)
    ax.set_ylim(bottom=-0.01, top=1.01)
    ax.set(xlabel='Time (s)')
    ax.set(ylabel='Nematic Order S')

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

    # save to h5py
    if 'filament/nematic_order' in params['data_filestream']:
        del params['data_filestream']['filament/nematic_order'] 
    params['data_filestream'].create_dataset('filament/nematic_order', data=S, dtype='f')

def PlotPolarOrder(FData, params, savepath):
    """ Plot the global polar order over time """

    print('Plotting Polar Order')
    # Polar order S
    P = calc_polar_order( FData.orientation_)

    # Display
    timeStep = FData.config_['time_snap']
    timeArray = timeStep * np.arange(FData.nframe_)

    fig,ax = plt.subplots()
    ax.plot(timeArray, P, 'ko', ms=2);
    ax.set_xlim(left=-0.01)
    ax.set_ylim(bottom=-1.01, top=1.01)
    ax.set(xlabel='Time (s)')
    ax.set(ylabel='Polar Order')

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

    # save to h5py
    if 'filament/polar_order' in params['data_filestream']:
        del params['data_filestream']['filament/polar_order']
    params['data_filestream'].create_dataset('filament/polar_order', data=P, dtype='f')

def PlotZOrder(FData, savepath):
    """ Plot the Z orientational order over time """

    print('Plotting Z Orientational Order')
    # Z order
    Z = calc_z_ordering( FData.orientation_)

    # Display
    timeStep = FData.config_['time_snap']
    timeArray = timeStep * np.arange(FData.nframe_)

    fig,ax = plt.subplots()
    ax.plot(timeArray, Z, 'ko', ms=2, label=r'$Z= < | u_{i,z} | >');
    ax.set_xlim(left=-0.01)
    ax.set_ylim(bottom=-0.01, top=1.01)
    ax.set(xlabel='Time (s)')
    ax.set(ylabel='Z Order')

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")

def PlotNematicAndPolarOrder(FData, savepath):
    """ Plot the global nematic order and polar order over time """

    print('Plotting Nematic and Polar Order')
    # Nematic order S
    S = calc_nematic_order( FData.orientation_)

    # Polar order S
    P = calc_polar_order( FData.orientation_)

    # Display
    timeStep = FData.config_['time_snap']
    timeArray = timeStep * np.arange(FData.nframe_)

    fig,ax = plt.subplots()
    ax.plot(timeArray, S, 'ro', ms=2, label='Nematic Order S');
    ax.plot(timeArray, P, 'bo', ms=2, label='Polar Order');
    ax.set_xlim(left=-0.01)
    ax.set_ylim(bottom=-1.01, top=1.01)
    ax.set(xlabel='Time (s)')
    ax.set(ylabel='Order Parameter')
    ax.legend()

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")

def calc_nematic_order(orient_array):
    """
    Calculates the nematic order S by computing the maximum eigenvalue
    of the nematic tensor Q
    Inputs: 
        orient_array : 3 x N x T (where N is number of filaments, T is number of frames)
    """

    # Number of frames 
    n_frames = orient_array.shape[2]
    S = np.zeros(n_frames)

    for jframe in np.arange(n_frames):

        # calculate Q tensor
        Q = calc_nematic_tensor(orient_array[:,:,jframe])

        # Find largest eigenvalue
        S[jframe] = np.sqrt(np.tensordot(Q, Q)*1.5)
        
    return S

# @njit
def calc_polar_order(orient_array):
    """
    Calculate the global polar order
    Inputs: 
        orient_array : 3 x N x T (where N is number of filaments, T is number of frames)
    """

    # Num filaments
    n_fil = orient_array.shape[1]

    # Number of frames 
    n_frames = orient_array.shape[2]
    
    P = np.zeros(n_frames)
    for jframe in np.arange(n_frames):

        # Initialize P vector
        Pvec = np.zeros(3)

        # Take a mean of all orientations
        for irow in np.arange(n_fil):
            Pvec += orient_array[:,irow,jframe]
        Pvec = Pvec / n_fil
        P[jframe] = np.linalg.norm( Pvec)

    return P

@njit
def calc_nematic_tensor( orient_array):
    """
    Calculates the nematic tensor Q
    Inputs: 
        orient_array : 3 x N (where N is number of filaments)
    """

    # Num filaments
    n_fil = orient_array.shape[1]

    # initialize Q tensor
    Q = np.zeros((3,3))

    # sum over all filaments, taking their outer products
    for irow in np.arange(n_fil):
        Q += np.outer( orient_array[:,irow], orient_array[:,irow])

    # complete mean calculation by dividing by number of filaments and subtracting identity.
    Q = Q/n_fil - np.identity(3)/3
    return Q

def calc_nematic_basis( orient_array):
    """
    Calculates the nematic basis vectors, with the first being the nematic director 
    Inputs: 
        orient_array : 3 x N (where N is number of filaments)
    """

    # Num filaments
    # n_fil = orient_array.shape[1]

    # Find Q tensor
    Q = calc_nematic_tensor(orient_array)

    # Find eigenvectors and eigenvalues
    eigval, eigvec = np.linalg.eig(Q)

    # Nematic director is the eignevector correponding to the largest eigenvalue
    # This is the new z axis
    idxMax = np.where( eigval == max(eigval) )[0][0]
    znew = eigvec[:,idxMax]

    # Find the other two vectors
    idxOther = [ii  for ii in range(3) if ii != idxMax]

    if np.dot(np.cross(eigvec[:,idxOther[0]],eigvec[:,idxOther[1]]),znew)>0.99:
        e1 = eigvec[:,idxOther[0]]
        e2 = eigvec[:,idxOther[1]]
    elif np.dot(np.cross(eigvec[:,idxOther[1]],eigvec[:,idxOther[0]]),znew)>0.99:
        e1 = eigvec[:,idxOther[1]]
        e2 = eigvec[:,idxOther[0]]
    else:
        raise Exception('Basis not found')

    # Define basis
    basis = np.zeros((3,3))
    basis[:,0] = e1
    basis[:,1] = e2
    basis[:,2] = znew
    return basis

# def calc_nematic_director_over_time( orient_array):
    # """
    # Calculates the nematic director over time 
    # Inputs: 
        # orient_array : 3 x N X T (where N is number of filaments, T is the number of frames)
    # """
    # nframe = orient_array.shape[2]
    # director = np.zeros( (3,nframe) )

    # for jframe in np.arange(nframe):

        # # Find nematic tensor
        # Q = calc_nematic_tensor( orient)

        # # Find nematic director
        # eigval, eigvec = np.linalg.eig(Q)

        # # Nematic director is the eignevector correponding to the largest eigenvalue
        # idxMax = np.where( eigval == max(eigval) )[0][0]
        # director[:,jframe] = eigvec[:,idxMax]

    

# @njit
def calc_z_ordering( orient_array):
    """
    Calculates the ordering in Z
    Inputs: 
        orient_array : 3 x N x T (where N is number of filaments, T is number of frames)
    """

    z_axis = np.array( [0.,0.,1.])

    # Num filaments
    n_fil = orient_array.shape[1]

    # # Number of frames 
    # n_frames = orient_array.shape[2]
    
    # # Initialize return array
    # Zorder = np.zeros(n_frames)

    # for jframe in np.arange(n_frames):

        # sum_all = 0
        # for idx in np.arange( n_fil):
            # sum_all += np.absolute( orient_array[:,idx,jframe].dot(z_axis) )

        # Zorder[jframe] = sum_all/n_fil

    # Calculate orientational order w.r.t axis
    orient_order = np.tensordot(orient_array, z_axis, axes=((0),(0)))
    Zorder = np.sum( np.absolute(orient_order), axis=0) / n_fil
    
    return Zorder

def calc_nematic_order_xyz(orient_array):
    """
    Calculates the nematic order S in xyz directions 
    Inputs: 
        orient_array : 3 x N x T (where N is number of filaments, T is number of frames)
    """

    # Number of frames 
    n_frames = orient_array.shape[2]
    S = np.zeros((3,n_frames))

    for jframe in np.arange(n_frames):
        ort = orient_array[:,:,jframe]

        for jdim in range(3):
            cos_angle_sq = ort[jdim,:]**2
            S[jdim,jframe] = 0.5*np.mean( 3*cos_angle_sq - 1)
        
    return S
