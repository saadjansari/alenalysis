import numpy as np
import os
import seaborn as sns
import pdb
import matplotlib.pyplot as plt

# PlotExtensileFilamentPairs{{{
def PlotExtensileFilamentPairs( FData, XData, params, savepath):

    print('Computing Extensile Filament Pairs...') 

    # frames
    frames = np.arange(0, FData.nframe_)

    # Loop over crosslinkers
    # For each crosslinker, computing the dot product of the bound filaments.
    # If negative, filaments are anti aligned. If positive, filaments are aligned
    N_aligned = np.zeros(len(frames))
    N_antialigned = np.zeros(len(frames))
    N_max = XData.nxlink_

    for jframe in frames:

        idx = np.where( (XData.link0_[:,jframe] >= 0) & (XData.link1_[:,jframe] >= 0) )
        if len(idx) == 0:
            continue
        o0 = FData.orientation_[:,XData.link0_[idx,jframe].astype('int'),jframe]
        o1 = FData.orientation_[:,XData.link1_[idx,jframe].astype('int'),jframe]
        dp = np.sum(o0*o1, axis=0)
        N_aligned[jframe] = np.sum( dp >= 0)
        N_antialigned[jframe] = np.sum( dp < 0)

    # time array
    timeStep = FData.config_['time_snap']
    timeArray = timeStep * np.arange(FData.nframe_)

    fig,ax = plt.subplots()
    ax.plot(timeArray, N_aligned/N_max, label='Aligned');
    ax.plot(timeArray, N_antialigned/N_max, label='Anti-aligned');
    ax.legend(frameon=False)
    ax.set_xlim(left=-0.01)
    ax.set_ylim(bottom=-0.01, top=1.01)
    ax.set(xlabel='Time (s)')
    ax.set(ylabel='Fraction of crosslinking\nmotor proteins')

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")

    # save to h5py
    if 'filament/time_array' in params['data_filestream']:
        del params['data_filestream']['filament/time_array']
    if 'filament/frac_aligned' in params['data_filestream']:
        del params['data_filestream']['filament/frac_aligned']
    if 'filament/frac_antialigned' in params['data_filestream']:
        del params['data_filestream']['filament/frac_antialigned']
    params['data_filestream'].create_dataset('filament/time_array', data=timeArray, dtype='f')
    params['data_filestream'].create_dataset('filament/frac_aligned', data=N_aligned/N_max, dtype='f')
    params['data_filestream'].create_dataset('filament/frac_antialigned', data=N_antialigned/N_max, dtype='f')
    plt.close()

# }}}
