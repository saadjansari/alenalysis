import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import seaborn as sns
from src.CalcOrderParameters import calc_nematic_basis
from src.calc_com_pbc import calc_com_pbc
from src.calc_pdist_pbc import pdist_pbc, pdist_pbc_xyz_max
from src.CalcOverlaps import minDistBetweenTwoFil
import pdb

def PlotPairDistribution(FData, params, savepath, frame=-1, window=5):

    print('Calculating 3D pair distribution function...')
    hist, bins = PairDistributionFunc(FData, frame=frame, window=window)

    fig,ax = plt.subplots(2,3, figsize=(12,6) )
    ax[0,0].imshow( np.mean(hist, axis=2), origin='lower', 
            extent=(bins[0][0], bins[0][-1], bins[1][0], bins[1][-1]),
            interpolation='bicubic')
    ax[0,0].set( xlabel=r'X ($\mu m$)',
            ylabel=r'Y ($\mu m$)',
            title='XY mean')

    ax[0,1].imshow( np.mean(hist, axis=0).T, origin='lower', 
            extent=(bins[1][0], bins[1][-1], bins[2][0], bins[2][-1]),
            interpolation='bicubic')
    ax[0,1].set( xlabel=r'Y ($\mu m$)',
            ylabel=r'Z ($\mu m$)',
            title='YZ mean')
    ax[0,2].imshow( np.mean(hist, axis=1).T, origin='lower', 
            extent=(bins[0][0], bins[0][-1], bins[2][0], bins[2][-1]),
            interpolation='bicubic')
    ax[0,2].set( xlabel=r'X ($\mu m$)',
            ylabel=r'Z ($\mu m$)',
            title='XZ mean')
    
    xc = int( hist.shape[0]/2)
    yc = int( hist.shape[1]/2)
    zc = int( hist.shape[2]/2)
    vxy = np.max(hist[:,:,zc])
    vyz = np.max(hist[xc,:,:])
    vxz = np.max(hist[:,yc,:])
    ax[1,0].imshow( hist[:,:,zc], origin='lower', 
            extent=(bins[0][0], bins[0][-1], bins[1][0], bins[1][-1]),
            interpolation='bicubic', vmin=0, vmax=vxy)
    ax[1,0].set( xlabel=r'X ($\mu m$)',
            ylabel=r'Y ($\mu m$)',
            title='XY center')
    ax[1,1].imshow( hist[xc,:,:].T, origin='lower', 
            extent=(bins[1][0], bins[1][-1], bins[2][0], bins[2][-1]),
            interpolation='bicubic', vmin=0, vmax=vyz)
    ax[1,1].set( xlabel=r'Y ($\mu m$)',
            ylabel=r'Z ($\mu m$)',
            title='YZ center')
    ax[1,2].imshow( hist[:,yc,:].T, origin='lower', 
            extent=(bins[0][0], bins[0][-1], bins[2][0], bins[2][-1]),
            interpolation='bicubic', vmin=0, vmax=vxz)
    ax[1,2].set_title('XZ center')
    ax[1,2].set( xlabel=r'X ($\mu m$)',
            ylabel=r'Z ($\mu m$)',
            title='XZ center')

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

def PairDistributionFunc(FData, frame=-1, window=5):

    # box_size
    bs = FData.config_['box_size']

    # Filament Diameter
    dd = 0.2*FData.config_['diameter_fil']
    dd_z = 10*dd

    # Bins
    rfil = 10*FData.config_['diameter_fil']
    rfil_z = 0.5
    # bins = [np.arange( -1*rfil, rfil+dd, dd).tolist(),
            # np.arange( -1*rfil, rfil+dd, dd).tolist(),
            # np.arange( -1*rfil_z,rfil_z+dd_z, dd_z).tolist()]
    bins = [ np.arange( -1*rfil + dd/2, rfil+ dd/2, dd).tolist(),
             np.arange( -1*rfil+ dd/2, rfil+ dd/2, dd).tolist(),
             np.arange( -1*rfil_z + dd_z/2, rfil_z+ dd_z/2, dd_z).tolist()]

    # frames
    frames = np.arange(FData.nframe_)
    frames = frames[frame-window:frame]

    # positions
    pos = FData.get_com()[:,:,frames]

    # average orientation
    ort = np.mean( FData.orientation_[:,:,frames], axis=-1)
    ort = ort / np.linalg.norm(ort, axis=0)

    # FIND COORD SYSTEM
    basis = np.zeros( (3,3) )
    # Z : nematic director
    basis[:,2] = calc_nematic_basis(ort)[:,-1]
    # Y: Z x X_lab
    basis[:,1] = np.cross( basis[:,2], np.array([1,0,0]) )
    # X: Y x Z
    basis[:,0] = np.cross( basis[:,1], basis[:,2])

    # Initialize histogram
    hist = np.zeros( (len(bins[0])-1, len(bins[1])-1, len(bins[2])-1 ) )
    for idx in np.arange(len(frames)):
        print(f'Frame {idx}')
        hist += PairDistributionFuncFrame( pos[:,:,idx], bins=bins, basis=basis)

    hist = hist/len(frames)
    return hist, bins

def PairDistributionFuncFrame(pos, bins=None, basis=None):
        
    # Check if basis is provided for transformation
    if basis is not None:
        pos = np.linalg.solve( basis, pos)

    # If bins aren't provided, define bins
    if bins is None:

        bins = [np.linspace(-1,1,100).tolist(),
                np.linspace(-1,1,100).tolist(),
                np.linspace(-1,1,100).tolist()]

    # Initialize histogram
    dat = np.zeros( (len(bins[0])-1, len(bins[1])-1, len(bins[2])-1 ) )

    # For each filament, transform to it's c.o.m frame
    for jfil in np.arange(pos.shape[-1]):
        # xyz = pos[:,np.arange(pos.shape[-1]) != jfil] - pos[:,jfil].reshape(3,1)
        xyz = pos- pos[:,jfil].reshape(3,1)

        # Only include filaments within the bins region
        idxKeep = ((np.abs(xyz[0,:])<bins[0][-1]) & (np.abs(xyz[1,:])<bins[1][-1]) & (np.abs(xyz[2,:])<bins[2][-1]) & (np.arange(pos.shape[-1]) != jfil))
        

        dat += np.histogramdd(xyz[:,idxKeep].T, bins=bins)[0]
        # pdb.set_trace()
    dat = dat / pos.shape[-1]
    
    return dat

def PlotCrosslinkedPairDist(FData, XData, savepath, window=10):

    # frames
    frames = np.arange(FData.nframe_-window, FData.nframe_)

    # Loop over crosslinkers
    # For each crosslinker, computing the dot product of the bound filaments.
    # If negative, filaments are anti aligned. If positive, filaments are aligned
    N_aligned = np.zeros(len(frames))
    N_antialigned = np.zeros(len(frames))
    N_max = XData.nxlink_

    dists_min = []
    dists_com = np.array([])
    for jframe in frames:

        idx = np.where( (XData.link0_[:,jframe] >= 0) & (XData.link1_[:,jframe] >= 0) )[0]
        if len(idx) == 0:
            continue
        idx1Fil = XData.link0_[idx,jframe].astype('int')
        idx2Fil = XData.link1_[idx,jframe].astype('int')
        p0 = FData.pos_minus_[:,idx1Fil,jframe]
        p1 = FData.pos_plus_[:,idx1Fil,jframe]
        q0 = FData.pos_minus_[:,idx2Fil,jframe]
        q1 = FData.pos_plus_[:,idx2Fil,jframe]
        for jpair in np.arange(len(idx)):
            dists_min.append( minDistBetweenTwoFil(p0[:,jpair], p1[:,jpair], 
                    q0[:,jpair], q1[:,jpair], FData.config_['box_size']) )
        dists_com = np.hstack( (dists_com, np.sqrt(np.sum(((p1+p0)/2 - (q0+q1)/2)**2, axis=0))  ) )

    bins = np.linspace(0,0.12,100)

    fig,ax = plt.subplots(1,1, figsize=(4,3))
    ax = sns.histplot( x=dists_com, y=dists_min, bins=(bins,bins), 
            cbar=True, cbar_kws=dict(label='Count'), ax=ax)
    ax.set(xlabel=r'C.O.M distance $(\mu m)$', 
            ylabel=r'Min distance $(\mu m)$', 
            title='Distance between crosslinked filaments')
    
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")

def Test():

    print('Testing: 3D pair distribution function...')

    hist, bins = test_aligned_state()

    fig,(ax0,ax1,ax2) = plt.subplots(1,3, figsize=(12,3) )
    ax0.imshow( np.mean(hist, axis=2), origin='lower', 
            extent=(bins[0][0], bins[0][-1], bins[1][0], bins[1][-1]))
    ax0.set_title('XY')
    ax1.imshow( np.mean(hist, axis=0).T, origin='lower', 
            extent=(bins[1][0], bins[1][-1], bins[2][0], bins[2][-1]))
    ax1.set_title('YZ')
    ax2.imshow( np.mean(hist, axis=1).T, origin='lower', 
            extent=(bins[0][0], bins[0][-1], bins[2][0], bins[2][-1]))
    ax2.set_title('XZ')
    plt.show()
    print('hello')
        
def test_aligned_state( Nx=10, Ny=10, Nz=5, spacing=1):

    x_vals = spacing*np.arange(Nx)
    y_vals = spacing*np.arange(Ny)
    z_vals = spacing*np.arange(Nz)
    
    pos = np.zeros( (3, Nx*Ny*Nz) )
    xv, yv, zv = np.meshgrid( x_vals, y_vals, z_vals)
    pos[0,:], pos[1,:], pos[2,:] = xv.flatten(), yv.flatten(), zv.flatten()

    # Bins
    xmax = np.max(x_vals)/2
    ymax = np.max(y_vals)/2
    zmax = np.max(z_vals)/2
    dd = 0.5*spacing
    bins = [ np.arange( -1*xmax + dd/2, xmax + dd/2, dd).tolist(),
             np.arange( -1*ymax + dd/2, ymax + dd/2, dd).tolist(),
             np.arange( -1*zmax + dd/2, zmax + dd/2, dd).tolist()]

    # FIND COORD SYSTEM
    basis = np.zeros( (3,3) )
    # Z : nematic director
    basis[:,2] = [0.0,0.1,0.9]
    basis[:,2] = basis[:,2] / np.linalg.norm( basis[:,2])
    # Y: Z x X_lab
    basis[:,1] = np.cross( basis[:,2], np.array([1,0,0]) )
    # X: Y x Z
    basis[:,0] = np.cross( basis[:,1], basis[:,2])

    hist = PairDistributionFuncFrame(pos, bins=bins, basis=basis)
    return hist, bins
    
# SampleFilamentLength {{{
def SampleFilamentLength(pos_plus, pos_minus, diameter):

    # diameter
    dd = diameter

    # Filament Lengths
    fil_len = np.linalg.norm(pos_plus - pos_minus, axis=0)
    fil_len_max = np.max(fil_len)
    
    # number of sampling points
    n_samples = 1*int(np.ceil(fil_len_max / dd) )

    # Break each filament into cylinders with diameter and height equal to the filament diameter
    pos_sampled = np.linspace(pos_minus, pos_plus, n_samples, axis=1).reshape((pos_minus.shape[0], -1), order='F')

    return pos_sampled 
# }}}
