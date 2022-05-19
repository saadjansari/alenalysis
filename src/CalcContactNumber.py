import numpy as np
from src.DataHandler import *
from src.write2vtk import add_array_to_vtk

# Plotting
def PlotContactNumberVsTime( FData, savepath, N=400):
    """ Plot contact number for each filament vs time """

    print('Plotting Contact Number Per Time Image ...')
    cNumber = FData.contact_number[:,-1*N:]

    fig,ax = plt.subplots()
    im = ax.imshow(cNumber, cmap='viridis', interpolation='nearest', 
            aspect='auto', vmin=-1, vmax=np.max(cNumber.flatten()) )
    plt.colorbar(im, ax=ax)
    ax.set(xlabel='Frame', ylabel='Filament', 
            title='Contact Number')
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

def PlotContactNumberHistogram( FData, savepath, N=400):
    """ Plot contact number histogram over all of time """

    print('Plotting Contact Number Histogram ...')
    cNumber = FData.contact_number[:,-1*N:]

    fig,ax = plt.subplots()
    ax.hist(cNumber.flatten(), bins=np.linspace(0,np.max(cNumber.flatten()),50))[-1]
    ax.set(yscale='log', xlabel='Contact Number', ylabel='Counts' )
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()
    
def WriteContactNumber2vtk( FData, params):

    cn_known = FData.contact_number
    cn = np.zeros(( FData.nfil_, FData.nframe_))
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
