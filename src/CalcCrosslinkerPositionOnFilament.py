import numpy as np
import os
import seaborn as sns
import pdb
import matplotlib.pyplot as plt

# PlotCrosslinkerPositionOnFilament {{{
def PlotCrosslinkerPositionOnFilament( FData, XData, params):
    """ Make a Density Plot for all frames"""

    print('Making Crosslinker Distribution On Filaments Plot...') 

    # Paths
    savepath = params['plot_path'] / 'crosslinker_position_on_filament.pdf'

    # frames
    frames = np.arange(0, FData.nframe_)

    # bins
    bins = np.linspace(-1,1,51)
    
    # filament lengths
    p_xyz = FData.pos_plus_-FData.pos_minus_
    for jdim in range(p_xyz.shape[0]):
        p_xyz[jdim,:,:][ p_xyz[jdim,:,:] < -0.5*FData.config_['box_size'][jdim] ] += FData.config_['box_size'][jdim]
        p_xyz[jdim,:,:][ p_xyz[jdim,:,:] > 0.5*FData.config_['box_size'][jdim] ] -= FData.config_['box_size'][jdim]
    fil_lens = np.linalg.norm( p_xyz, axis=0)
    
    # Find dist of xlinker heads from filament minus end 
    D0 = np.zeros( (len(bins)-1, FData.nframe_))
    D1 = np.zeros( (len(bins)-1, FData.nframe_))
    
    # for each time
    for jt in frames:
        link0 = XData.link0_[:,jt].astype(np.int64)
        link1 = XData.link1_[:,jt].astype(np.int64)

        # Inidces of bound heads
        idx0 = np.where(link0 > -1)[0]
        idx1 = np.where(link1 > -1)[0]

        # distanes from filament minus end for each bound xlinker head
        # scale the distances so that d0=+1 (at the plus end), d0=0 (center), d0=-1 (at the minus end)
        # Ensure head is in the same periodic image as the filament com
        if len(idx0) > 0:
            d0_xyz = XData.pos_minus_[:,idx0,jt] - FData.pos_minus_[:,link0[idx0],jt]
            for jdim in range(d0_xyz.shape[0]):
                d0_xyz[jdim,:][ d0_xyz[jdim,:] < -0.5*FData.config_['box_size'][jdim] ] += FData.config_['box_size'][jdim]
                d0_xyz[jdim,:][ d0_xyz[jdim,:] > 0.5*FData.config_['box_size'][jdim] ] -= FData.config_['box_size'][jdim]
            d0 = np.linalg.norm( d0_xyz, axis=0)
            d0 = 2*(d0 / fil_lens[link0[idx0],jt])-1
            D0[:,jt],_ = np.histogram( d0, bins=bins, density=True)
            
        if len(idx1) > 0:
            d1_xyz = XData.pos_plus_[:,idx1,jt] - FData.pos_minus_[:,link1[idx1],jt]
            for jdim in range(d1_xyz.shape[0]):
                d1_xyz[jdim,:][ d1_xyz[jdim,:] < -0.5*FData.config_['box_size'][jdim] ] += FData.config_['box_size'][jdim]
                d1_xyz[jdim,:][ d1_xyz[jdim,:] > 0.5*FData.config_['box_size'][jdim] ] -= FData.config_['box_size'][jdim]
            d1 = np.linalg.norm( d1_xyz, axis=0)
            # pdb.set_trace()
            d1 = 2*(d1 / fil_lens[link1[idx1],jt])-1
            D1[:,jt],_ = np.histogram( d1, bins=bins, density=True)
    
    # Normalize counts by total number of crosslinkers
    # D0 = D0 / XData.link0_.shape[0]
    # D1 = D1 / XData.link0_.shape[0]

    fig,ax = plt.subplots(1,2, sharey=True)

    # Head 0
    im = ax[0].imshow(D0.T, cmap='viridis', interpolation='gaussian', 
            aspect='auto', vmin=0, vmax=1,origin='upper',
            extent=[-1, 1, frames[-1], frames[0]])
    ax[0].set(title='Head 0', ylabel='Frame', xlabel='Position on filament')
    ax[0].set_yticks( np.arange(0,frames[-1],500))
    ax[0].set_xticks( [-1,0,1])
    # plt.colorbar(im, ax=ax[0])

    # Head 1
    im = ax[1].imshow(D1.T, cmap='viridis', interpolation='gaussian', 
            aspect='auto', vmin=0, vmax=1, origin='upper',
            extent=[-1, 1, frames[-1], frames[0]])
    plt.colorbar(im, ax=ax[1], label='PDF')
    ax[1].set(title='Head 1', xlabel='Position on filament')
    ax[1].set_xticks( [-1,0,1])
    plt.tight_layout()

    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

# }}}

# PlotCrosslinkerLength
def PlotCrosslinkerLength( XData, params, savepath):

    dt = XData.config_['time_snap']
    times = dt*np.arange(XData.nframe_)
    
    lens = XData.pos_plus_-XData.pos_minus_
    lens = np.linalg.norm(lens, axis=0)
    lens[ lens == 0] = np.NaN

    lens_mu = np.nanmean(lens, axis=0)
    lens_std = np.nanstd(lens, axis=0)

    fig, ax = plt.subplots()
    ax.errorbar(times[::10], lens_mu[::10], yerr=lens_std[::10], 
            marker='.', ms=1, mew=0.5, mfc="None", alpha=0.5,
            lw=1, linestyle = 'None', elinewidth=0.5, capsize=1,
            )

    # Labels
    ax.set(ylabel=r'Length / $\mu m$', xlabel='Time / s')
    ax.set_ylim(bottom=0.0)

    # Save plot
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


