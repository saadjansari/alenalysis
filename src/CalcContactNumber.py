from numba import njit
import numpy as np
import scipy.spatial.ckdtree
from scipy.spatial.distance import pdist, squareform
import src.decorators
from src.DataHandler import *
from src.write2vtk import add_array_to_vtk
from pathlib import Path
# import decorators
# from DataHandler import *
# from write2vtk import add_array_to_vtk

# Plotting
def PlotContactNumberVsTime( FData, savepath, N=100):
    """ Plot contact number for each filament vs time """

    print('Plotting Contact Number Per Time Image ...')
    cNumber = calc_contact_number(FData.min_dist_)

    fig,ax = plt.subplots()
    im = ax.imshow(cNumber, cmap='viridis', interpolation='nearest', 
            aspect='auto', vmin=-1, vmax=np.max(cNumber.flatten()) )
    plt.colorbar(im, ax=ax)
    ax.set(xlabel='Frame', ylabel='Filament', 
            title='Contact Number')
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

def PlotContactNumberHistogram( FData, savepath, N=100):
    """ Plot contact number histogram over all of time """

    print('Plotting Contact Number Histogram ...')
    cNumber = calc_contact_number(FData.min_dist_)

    fig,ax = plt.subplots()
    ax.hist(cNumber.flatten(), bins=np.linspace(0,np.max(cNumber.flatten()),50))[-1]
    ax.set(yscale='log', xlabel='Contact Number', ylabel='Counts' )
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()
    
def WriteContactNumber2vtk( FData, params):

    cn = np.zeros(( FData.nfil_, FData.nframe_))
    cn_known = calc_contact_number(FData.min_dist_)
    cn[:, -1*cn_known.shape[1]:] = cn_known
    add_array_to_vtk(cn, 'ContactNumber', params['sim_path'])

# @src.decorators.timer
def calc_contact_number(md):
    # Calculate the contact number for each filament

    contactNumber = np.zeros( (md.shape[0], md.shape[-1]) )
    
    for jframe in range(0,md.shape[-1]):
        # print('Frame = {0}/{1}'.format(jframe,n_frames), end='\r', flush=True )
        contactNumber[:,jframe] = np.sum( np.exp(-1*(md[:,:,jframe]**2)), axis=1)
    return contactNumber

# CalcSaveContactNumber {{{
def CalcSaveContactNumber(FData, simpath):

    # Define mindist file path
    cn_path = simpath / 'contact_number.npy'

    # Check if data already exists. If yes, load it
    if Path.exists(cn_path):
        with open(str(cn_path), 'rb') as f:
            CN = np.load(f)
        
            # how many new frames to compute this for
            n_new = int(FData.nframe_ - CN.shape[-1])
            print('Computing contact number for {0} frames'.format(n_new))

            if n_new > 0:
                CN_new = np.zeros( (FData.nfil_, n_new) )
                CN = np.concatenate( (CN, CN_new), axis=-1)
            else:
                return CN 
    else:
        # New frames
        n_new = int(FData.nframe_)
        CN = np.zeros((FData.nfil_, n_new))

    # Process unprocessed frames
    for cframe in range(FData.nframe_-n_new,FData.nframe_):

        print('Frame = {0}/{1}'.format(cframe,FData.nframe_), 
                end='\r', flush=True )
    
        # overlap matrix (normalized by filament diameter)
        CN[:,cframe] = np.sum( np.exp(-1*(FData.MD[:,:,cframe]**2)), axis=1)

    print('Frame = {0}/{0}'.format(FData.nframe_))

    # Save contact number data
    with open(str(cn_path), 'wb') as f:
        np.save(f, CN)

    return CN 
# }}}

