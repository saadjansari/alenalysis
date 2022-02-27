import numpy as np
import random
from matplotlib import pyplot as plt
# import tidynamics
import pdb

# Plotting
def PlotMSD(FData, savepath):
    """ Plot the ensemble-averaged MSD"""

    print('Plotting ensemble-averaged MSD')

    # Unfolded trajectories
    pos_unfolded = FData.unfold_trajectories('com')

    # MSD
    MSD = calc_msd_fft( pos_unfolded)

    # Display
    timeStep = FData.config_['time_snap']
    timeArray = timeStep * np.arange(FData.nframe_)

    fig,ax = plt.subplots()
    ax.plot(timeArray, np.mean(MSD, axis=0), color='blue', lw=2)
    ax.fill_between(timeArray,
                    np.mean(MSD, axis=0) - np.std(MSD, axis=0)/np.sqrt(MSD.shape[0]),
                    np.mean(MSD, axis=0) + np.std(MSD, axis=0)/np.sqrt(MSD.shape[0]),
                    color='blue', alpha=0.2)
    ax.set_xlim(left=-0.01)
    ax.set_ylim(bottom=-0.01)
    ax.set(xlabel='Lag time / s')
    ax.set(ylabel=r'MSD / $\mu m^2$')

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

def autocorrFFT(x):
    N=len(x)
    F = np.fft.fft(x, n=2*N)  #2*N because of zero-padding
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res= (res[:N]).real   #now we have the autocorrelation in convention B
    n=N*np.ones(N)-np.arange(0,N) #divide res(m) by (N-m)
    return res/n #this is the autocorrelation in convention A

def msd_fft(r):
    """Compute the msd using fft for a single trajectory
    Note: trajectories must be unfolded prior to computation.
    Parameters
    ----------
    r : Numpy Array (T x 3)
    Returns
    -------
    msd: numpy array 1 x T 
    """
    N=len(r)
    D=np.square(r).sum(axis=1) 
    D=np.append(D,0) 
    S2=sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    Q=2*D.sum()
    S1=np.zeros(N)
    for m in range(N):
        Q=Q-D[m-1]-D[N-m]
        S1[m]=Q/(N-m)
    return S1-2*S2

def calc_msd_fft(pos):
    """Compute the msd using fft for each trajectory
    Note: trajectories must be unfolded prior to computation.
    Parameters
    ----------
    pos : Numpy Array (3 x N x T)
    Returns
    -------
    msd: numpy array N x T 
    """
    # Initiliaze msd array
    msd = np.zeros( (pos.shape[1], pos.shape[2]) )
    
    # Loop over trajectories
    for jtraj in range( pos.shape[1] ):
        msd[jtraj, :] = msd_fft(pos[:,jtraj,:].transpose())
        # msd1 = msd_fft(pos[:,jtraj,:].transpose())
        # msd[jtraj,:] = tidynamics.msd(pos[:,jtraj,:].transpose())
        # pdb.set_trace()
        # plt.plot(msd1,color='r'); plt.plot(msd2,color='b'); plt.show()
    return msd

# def calc_msd_brute(pos):

