from numba import njit
import numpy as np
import pdb
import scipy.spatial.ckdtree
from scipy.ndimage import convolve as conv
from src.DataHandler import *
from src.CalcContactNumber import calc_contact_number
from src.write2vtk import add_array_to_vtk
import src.decorators
from pathlib import Path
from matplotlib import colors

def PlotLocalPolarOrderVsTime( FData, savepath):
    """ Plot local polar order for each filament vs time """

    print('Plotting Local Polar Order Per Time Image (via KDtree)...')
    PolarOrder = FData.local_polar_order

    fig,ax = plt.subplots()

    im = ax.imshow(PolarOrder, cmap='viridis', interpolation='nearest', 
            aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set(xlabel='Frame', ylabel='Filament', 
            title='Local Polar Order')

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

def PlotLocalPolarOrderHistogram( FData, savepath, N=300):
    """ Plot local polar order histogram over all of time (KDtree) """

    print('Plotting Local Polar Order Histogram (via KDtree)...')
    PolarOrder = FData.local_polar_order[:,-1*N]

    fig,ax = plt.subplots()
    ax.hist(PolarOrder.flatten(), bins=np.linspace(-1,1,50))[-1]
    ax.set(yscale='log', xlabel='Local Polar Order KD', ylabel='Counts' )
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

def PlotLocalNematicOrderVsTime( FData, savepath):
    """ Plot local nematic order for each filament vs time """

    print('Plotting Local Nematic Order Per Time Image (via KDtree)...')
    SOrder = FData.local_nematic_order

    fig,ax = plt.subplots()

    im = ax.imshow(SOrder, cmap='viridis', interpolation='nearest', 
            aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set(xlabel='Frame', ylabel='Filament', 
            title='Local Nematic Order')

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

def PlotLocalNematicOrderHistogram( FData, savepath, N=300):
    """ Plot local nematic order histogram over all of time (KDtree) """

    print('Plotting Local Nematic Order Histogram (via KDtree)...')
    SOrder = FData.local_nematic_order[:,-1*N]

    fig,ax = plt.subplots()
    ax.hist(SOrder.flatten(), bins=np.linspace(0,1,50))[-1]
    ax.set(yscale='log', xlabel='Local Nematic Order', ylabel='Counts' )
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

def PlotLocalNematicOrderContactNumberHistogram( FData, savepath, N=300):
    """ Plot local nematic order vs contact number 2D histogram over all of time """

    print('Plotting 2D hist: Local Nematic Order and Contact number...')
    SOrder = FData.local_nematic_order[:,-1*N:].flatten()
    Cnumber = FData.contact_number[:,-1*N:].flatten()

    # evaluate on a regular grid
    xgrid = np.linspace(0,1,100)
    ygrid = np.linspace(np.min(Cnumber),np.max(Cnumber), 100)

    # Plot the result as an image
    fig,ax = plt.subplots()
    im = ax.hist2d(SOrder, Cnumber,
           bins = [xgrid, ygrid], density=True,
           cmap ="viridis")
    fig.colorbar(im[3], ax=ax, label="Density")
    ax.set(xlabel='Local nematic order', ylabel='Contact number')
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

def PlotLocalPolarOrderContactNumberHistogram( FData, savepath, N=300):
    """ Plot local polar order vs contact number 2D histogram over all of time """

    print('Plotting 2D hist: Local polar order and contact number...')
    POrder = FData.local_polar_order[:,-1*N:].flatten()
    Cnumber = FData.contact_number[:,-1*N:].flatten()

    # evaluate on a regular grid
    xgrid = np.linspace(-1,1,100)
    ygrid = np.linspace(np.min(Cnumber),np.max(Cnumber), 100)

    # Plot the result as an image
    fig,ax = plt.subplots()
    im = ax.hist2d(POrder, Cnumber,
           bins = [xgrid, ygrid], density=True,
           cmap ="viridis")
    fig.colorbar(im[3], ax=ax, label="Density")
    ax.set(xlabel='Local polar order', ylabel='Contact number')
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

def PlotLocalNematicOrderLocalPolarOrderHistogram( FData, savepath, N=300):
    """ Plot local nematic order vs local polar number 2D histogram over all of time """

    print('Plotting 2D hist: Local Nematic Order and local polar order...')
    POrder = FData.local_polar_order[:,-1*N:].flatten()
    SOrder = FData.local_nematic_order[:,-1*N:].flatten()

    # evaluate on a regular grid
    xgrid = np.linspace(0,1,100)
    ygrid = np.linspace(-1,1,100)

    # Plot the result as an image
    fig,ax = plt.subplots()
    im = ax.hist2d(SOrder, POrder,
           bins = [xgrid, ygrid], density=True,
           cmap ="viridis")
    fig.colorbar(im[3], ax=ax, label="Density")
    ax.set(xlabel='Local nematic order', ylabel='Local polar order')
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

def WriteLocalOrder2vtk( FData, params):

    lpo = np.zeros( (FData.nfil_, FData.nframe_) )
    lpo_known = FData.local_polar_order
    lpo[:, -1*lpo_known.shape[1]:] = lpo_known
    add_array_to_vtk(lpo, 'LPO', params['sim_path'])

    lno = np.zeros( (FData.nfil_, FData.nframe_) )
    lno_known = FData.local_nematic_order
    lno[:, -1*lno_known.shape[1]:] = lno_known
    add_array_to_vtk(lno, 'LNO', params['sim_path'])

# calc_local_order {{{
def calc_local_order( orients, min_dist):
    """
    Calculate the local polar order via KDtree
    Inputs: 
        orient_array : 3 x N x T 
        min_dist : N x N x T (minimum distance matrix between filament pairs)
    """
    # Number of frames 
    n_frames = orients.shape[2]

    # local polar order array
    Porder = np.zeros( (orients.shape[1], orients.shape[2]) )
    Sorder = np.zeros( (orients.shape[1], orients.shape[2]) )

    for jframe in range(0,n_frames):
        print('Frame = {0}/{1}'.format(jframe,n_frames-1) )
        Porder[:,jframe], Sorder[:,jframe] = calc_local_order_frame( 
                orients[:,:,jframe], 
                min_dist[:,:,jframe])

    # Smooth the orders (T = 5)
    convKernel = np.ones(10)/10
    for jfil in np.arange(Porder.shape[0]):
        Porder[jfil,:] = conv(Porder[jfil,:], convKernel, mode='mirror')
        Sorder[jfil,:] = conv(Sorder[jfil,:], convKernel, mode='mirror')
    return Porder, Sorder
# }}}

# calc_local_order_frame {{{
@src.decorators.timer
# @decorators.timer
def calc_local_order_frame( orients, min_dist):
    """
    Calculate the local polar order via min dist
    Inputs: 
        orient_array : 3 x N 
        min_dist : N x N
    """

    Porder = np.zeros( orients.shape[1])
    Sorder = np.zeros( orients.shape[1])

    # # Q tensor for all filaments
    # Q_all = np.diagonal( np.tensordot( orients[:,:], orients[:,:], axes=0), axis1=1, axis2=3)

    # Cos theta^2 of all filaments (N x N)
    cosTheta2 = np.tensordot(orients,orients,axes=((0),(0)))**2

    # get contact number for this frame
    contactNumber = np.sum( np.exp( -1*( min_dist[:,:]**2)), axis=1)

    # dist gaussian factor
    g_factor = np.exp( -1*min_dist**2 )

    for idx in np.arange( orients.shape[1]):

        # orientation of reference filament
        o1 = orients[:,idx].reshape(-1,1)

        # indices of other filaments
        idx_rest = np.delete( np.arange(orients.shape[1]), idx)

        # local polar order
        Porder[idx] = np.sum( np.sum( o1*orients[:,idx_rest], axis=0)*g_factor[idx,idx_rest]) / contactNumber[idx]
        
        # # Average weighed Q tensor
        # Q_local = np.average( Q_all[:,:,idx_rest], axis=2, weights=g_factor[idx,idx_rest]/contactNumber[idx]) - np.identity(3)/3
        # pdb.set_trace()

        # # Local Nematic Order
        # Sorder[idx] = np.sqrt(np.tensordot(Q_local, Q_local)*1.5)
        
        # Local Nematic Order
        Sorder[idx] = 0.5*np.sum( (3*cosTheta2[idx,idx_rest] - 1)*g_factor[idx,idx_rest] ) / contactNumber[idx]

    return Porder, Sorder
# }}}
