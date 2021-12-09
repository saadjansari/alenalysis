from numba import njit
import numpy as np
from matplotlib import pyplot as plt
import decorators
import pdb


def PlotMobilityFilamentVsTime( FData, savepath):
    """ Plot mobility for each filament vs time """

    print('Plotting Number of Crosslinkers Per Filament Per Time Image...')

    # Unfolded center positions
    pos_unfolded = FData.unfold_trajectories('com')

    # Mobility
    mobi = calc_mobility(pos_unfolded)
    pdb.set_trace()

    # Display image
    fig,ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(mobi, cmap='viridis', interpolation='nearest', aspect=0.3)
    plt.colorbar(im, ax=ax)
    ax.set(xlabel='Frame', ylabel='Filament', 
            title=r'Mobility / $\mu m^2$')

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

@decorators.timer
def calc_mobility(pos,windowSize=10):
    # calculate mobility 3D: squared displacement

    nfil = pos.shape[1]
    ntime = pos.shape[2]

    convKernel = np.ones(windowSize)/windowSize
    cumDisp = np.zeros_like(pos[-1,:,:])

    for jfil in np.arange(nfil):
        pos_unfolded = pos[:,jfil,:]
        pos2 = np.linalg.norm( np.diff(pos_unfolded, axis=1), axis=0)
        cumDisp[jfil,1:] = np.convolve(pos2, convKernel, 'same')
    cumDisp[:,0] = cumDisp[:,1]
    return cumDisp

