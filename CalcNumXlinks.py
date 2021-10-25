from numba import njit
import numpy as np
import decorators
from DataHandler import *

# Plotting
def PlotStateCounts( FData, XData, savepath):
    """ Plot total number of xlinkers in each crosslinker state vs time """

    print('Plotting Crosslinker State Counts...')
    # Number of xlinks
    nxf,nxs,nxd = calc_num_xlink(XData.link0_, XData.link1_)

    # Display
    timeStep = FData.time_snap_
    timeArray = timeStep * np.arange(FData.nframe_)

    fig,ax = plt.subplots()
    # compare the total number of singly bound heads and free heads
    ax.plot(timeArray, nxf, 'ko', label='Free', ms=1, alpha=0.7);
    ax.plot(timeArray, nxs, 'go', label='SinglyBound', ms=1, alpha=0.7);
    ax.plot(timeArray, nxd, 'bo', label='DoublyBound', ms=1, alpha=0.7);
    ax.set_xlim(left=-0.01)
    ax.set_ylim(bottom=-0.2)
    ax.set(xlabel='Time (s)')
    ax.set_ylabel('Num crosslinkers\nin state')
    ax.legend()

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    

def PlotXlinkPerFilamentVsTime( FData, XData, savepath):
    """ Plot number of xlinkers for each filament vs time """

    print('Plotting Number of Crosslinkers Per Filament Per Time Image...')
    nxs,nxd = calc_num_xlink_filament(XData.link0_, XData.link1_, FData.nfil_)

    fig,axs = plt.subplots(1,2, figsize=(12,6), sharey=True)

    # Singly
    im = axs[0].imshow(nxs, cmap='viridis', interpolation='nearest', aspect=0.3)
    plt.colorbar(im, ax=axs[0])
    axs[0].set(xlabel='Frame', ylabel='Filament', 
            title='Number of singly-bound \n xlinks')

    # Doubly 
    im = axs[1].imshow(nxd, cmap='viridis', interpolation='nearest', aspect=0.3)
    plt.colorbar(im, ax=axs[1])
    axs[1].set(xlabel='Frame', ylabel='Filament', 
            title='Number of doubly-bound \n xlinks')
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

def PlotXlinkPerFilament( FData, XData, savepath):
    """ Plot number of xlinkers for each filament """

    print('Plotting Crosslinker Per Filament Histogram...')
    nxs,nxd = calc_num_xlink_filament(XData.link0_, XData.link1_, FData.nfil_)

    fig,axs = plt.subplots(1, 3, figsize=(12,3))
    axs[0].hist(nxs.flatten(), bins=np.arange(1+np.max(nxs.flatten())))[-1]
    axs[0].set(yscale='log', xlabel='Number of Crosslinkers', ylabel='Counts', title='Singly-Bound')
    axs[1].hist(nxd.flatten(), bins=np.arange(1+np.max(nxd.flatten())))[-1]
    axs[1].set(yscale='log', xlabel='Number of Crosslinkers', ylabel='Counts', title='Doubly-Bound')
    axs[2].hist(nxs.flatten()+nxd.flatten(), 
            bins=np.arange(1+np.max(nxd.flatten()+nxs.flatten())))[-1]
    axs[2].set(yscale='log', xlabel='Number of Crosslinkers', ylabel='Counts', title='All-Bound')
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

def PlotXlinkPerFilamentVsTimeMax( FData, XData, savepath, numPlot=5):
    """ Plot number of xlinkers for each filament vs time for the xlinkers with max val"""

    print('Plotting Crosslinker Per Filament Vs Time for heavily saturated filaments...')
    nxs,nxd = calc_num_xlink_filament(XData.link0_, XData.link1_, FData.nfil_)
    nx_all = nxs+nxd

    # Indices of filaments with max bound crosslinkers
    nx_max = np.max( nx_all, axis=1)
    inds = np.argpartition(nx_max, -numPlot)[-numPlot:]

    # Display
    timeStep = FData.time_snap_
    timeArray = timeStep * np.arange(FData.nframe_)

    fig,ax = plt.subplots(figsize=(8,3))
    for ind in inds:
        ax.plot(timeArray, nx_all[ind,:], 'o', label=ind, ms=1, alpha=0.7);

    ax.set(xlabel='Time (s)')
    ax.set_ylabel('Num bound crosslinkers')
    ax.legend()
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

def CalcNumXlink_Filament_Singly( FData, XData):
    nxs,_ = calc_num_xlink_filament(XData.link0, XData.link1, FData.nfil_)
    return nxs
    
def CalcNumXlink_Filament_Doubly( FData, XData):
    _,nxd = calc_num_xlink_filament(XData.link0, XData.link1, FData.nfil_)
    return nxd

def CalcNumXlink_Free( FData, XData):
    nxf,_,_ = calc_num_xlink(XData.link0, XData.link1)
    return nxf

def CalcNumXlink_Singly( FData, XData):
    _nxs,_ = calc_num_xlink(XData.link0, XData.link1)
    return nxs
    
def CalcNumXlink_Doubly( FData, XData):
    _,_,nxd = calc_num_xlink(XData.link0, XData.link1)
    return nxd

@decorators.timer
@njit
def calc_num_xlink_filament(link0,link1, n_fil):
    # calculate the num of singly, doubly bound xlinkers for each filament

    n_xlinker = np.shape(link0)[0]
    n_time = np.shape(link0)[1]
    nxd_per_fil = np.zeros( (n_fil, n_time) )
    nxs_per_fil = np.zeros( (n_fil, n_time) )

    for jt in np.arange(n_time):
        for jx in np.arange(n_xlinker):
            idx0 = np.int(link0[jx,jt])
            idx1 = np.int(link1[jx,jt])
            if idx0 > -1 and idx1 > -1:
                nxd_per_fil[idx0,jt]+=1
                nxd_per_fil[idx1,jt]+=1
            elif idx0 > -1:
                nxs_per_fil[idx0,jt]+=1
            elif idx1 > -1:
                nxs_per_fil[idx1,jt]+=1
    return nxs_per_fil, nxd_per_fil

@decorators.timer
@njit
def calc_num_xlink(link0,link1):
    # calculate the num of singly, doubly bound xlinker crosslinkers 

    n_xlinker = np.shape(link0)[0]
    n_time = np.shape(link0)[1]
    nxd = np.zeros(n_time)
    nxs = np.zeros(n_time)
    nxf = np.zeros(n_time)

    for jt in np.arange(n_time):
        for jx in np.arange(n_xlinker):
            idx0 = np.int(link0[jx,jt])
            idx1 = np.int(link1[jx,jt])
            if idx0 > -1 and idx1 > -1:
                nxd[jt]+=1
            elif idx0 > -1 or idx1 > -1:
                nxs[jt]+=1
            elif idx0 == -1 and idx1 == -1:
                nxf[jt]+=1
            else:
                raise ValueError('how in the')
    return nxf,nxs,nxd
