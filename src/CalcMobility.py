from numba import njit
import numpy as np
from matplotlib import pyplot as plt
import src.decorators
import pdb
from scipy.ndimage import convolve as conv


def PlotMobilityFilamentVsTime( FData, savepath):
    """ Plot mobility for each filament vs time """

    print('Plotting Filament mobility per time...')

    # Unfolded center positions
    pos_unfolded = FData.unfold_trajectories('com')

    # Mobility
    mobi = calc_mobility(pos_unfolded)

    # Display image
    fig,ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(mobi, cmap='viridis', interpolation='nearest', aspect=0.3)
    plt.colorbar(im, ax=ax)
    ax.set(xlabel='Frame', ylabel='Filament', 
            title=r'Mobility / $\mu m^2$')

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

def PlotMobilityCrosslinkerVsTime( XData, savepath):
    """ Plot mobility for each crosslinker vs time """

    print('Plotting Crosslinker mobility per time...')

    # Unfolded center positions
    pos_unfolded = XData.unfold_trajectories('com')

    # Mobility
    mobi = calc_mobility(pos_unfolded)

    # Display image
    fig,ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(mobi, cmap='viridis', interpolation='nearest', aspect=0.3)
    plt.colorbar(im, ax=ax)
    ax.set(xlabel='Frame', ylabel='Crosslinker', 
            title=r'Mobility / $\mu m^2$')

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

def PlotMobilityCrosslinkerHist( XData, savepath, N=500):
    """ Plot mobility for each crosslinker histogram"""

    print('Plotting Crosslinker mobility per time...')

    # Unfolded center positions
    pos_unfolded = XData.unfold_trajectories('com')[:,:,-1*N:]

    # Mobility
    mobi = calc_mobility(pos_unfolded)

    fig,ax = plt.subplots(1, 1, figsize=(4,3))
    ax.hist(mobi.flatten(), bins=50)[-1]
    ax.set(yscale='log', xlabel=r'Mobility / $\mu m^2$', ylabel='Counts')
    ax.set_xlim(right=0.5)
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

@src.decorators.timer
def calc_mobility(pos,windowSize=10):
    # calculate mobility 3D: squared displacement

    nfil = pos.shape[1]
    ntime = pos.shape[2]

    convKernel = np.ones(windowSize)/windowSize
    cumDisp = np.zeros_like(pos[-1,:,:])

    for jfil in np.arange(nfil):
        pos_unfolded = pos[:,jfil,1:]
        pos2 = np.linalg.norm( np.diff(pos_unfolded, axis=1), axis=0)
        cumDisp[jfil,2:] = conv(pos2, convKernel, mode='constant', cval=np.mean(pos2.flatten()) )
    cumDisp[:,0] = cumDisp[:,2]
    cumDisp[:,1] = cumDisp[:,2]
    return cumDisp

