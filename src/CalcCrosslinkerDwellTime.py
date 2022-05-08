import numpy as np
import os
import seaborn as sns
import pdb
import matplotlib.pyplot as plt
from itertools import groupby

# PlotCrosslinkerDwellTimes{{{
def PlotCrosslinkerDwellTime( XData, savepath, N=500):
    """ Make a Histogram for all last N frames"""

    # frames
    frames = np.arange(XData.nframe_-N, XData.nframe_)

    # bound heads
    head0 = XData.link0_[:,-1*N:].astype(np.int)
    head1 = XData.link1_[:,-1*N:].astype(np.int)
    
    frames_bound = []
    # For each head, calculate length of repeated values
    for jx in np.arange(head0.shape[0]):
        for key,g in groupby(head0[jx,:]):
            # keys.append(key)
            if key != -1:
                frames_bound.append( 0.05*len(list(g)))
    for jx in np.arange(head1.shape[0]):
        for key,g in groupby(head1[jx,:]):
            # keys.append(key)
            if key != -1:
                frames_bound.append( 0.05*len(list(g)))

    fig,ax = plt.subplots(1, 1, figsize=(4,3))
    ax.hist(frames_bound, bins=20, density=True)[-1]
    # ax.set(yscale='log', xlabel=r'Mobility / $\mu m^2$', ylabel='Counts')
    ax.set(xlabel='Dwell time / s', ylabel='Counts')
    ax.set_xlim(right=5.0)
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()
# }}}

