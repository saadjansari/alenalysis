from numba import njit
import numpy as np
from matplotlib import pyplot as plt
import src.decorators
from src.CalcMobility import calc_mobility
from src.CalcNumXlinks import calc_num_xlink_filament
import pdb
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import seaborn as sns
from src.CalcOrderParameters import calc_nematic_tensor
from src.CalcMSD import calc_msd_fft
from scipy.spatial.distance import pdist, squareform
from src.calc_gyration_tensor import calc_gyration_tensor3d_pbc
from src.CalcPackingFraction import calc_local_packing_fraction_frame

def PlotFilamentCondensation(FData, XData, params):
    """ Plot the filament condensation """
    
    # Filament clusters
    print('Filaments condensation...') 
    labels, ratio = condensed_via_xlinks_and_mobility(XData, FData, save=True, 
            savepath=params['plot_path'] / 'condensed_filament_positions.pdf',
            datapath=params['data_filestream'])

    # Plot ratio of condensed filaments over time
    PlotRatioCondensedFilaments(ratio, params['plot_path'] / 'condensed_filament_ratio.pdf')

    # Condensed MSD
    pos_unfolded = FData.unfold_trajectories('com')
    condensed_msd_diffusion(pos_unfolded, labels, save=True, 
            savepath=params['plot_path'] / 'condensed_filament_msd.pdf',
            datapath=params['data_filestream'])

    # Condensed/Vapor PF
    packing_fraction( 
            FData.pos_minus_[:,:,-1],
            FData.pos_plus_[:,:,-1],
            FData.config_,
            labels,
            save=True,
            savepath=params['plot_path'] / 'filaments_pf.pdf',
            datapath=params['data_filestream'])
            
    # # Plot time averaged label
    # PlotTimeAvgLabelHist(labels, params['plot_path'] / 'condensed_filament_time_avg_label.pdf')

def PlotFilamentClusters(FData, params, N=50):
    """ Plot the filament plus-end clusters """
    
    # Get last N frames
    frames = np.arange(FData.nframe_-N,FData.nframe_)

    # Filament plus-end clusters
    print('Filaments plus-end clusters...') 
    pos = FData.pos_plus_
    labels = cluster_via_dbscan(pos[:,:,frames], FData.config_['box_size'],
            save=True, savepath=params['plot_path'] / 'cluster_filament_dbscan.pdf')

    # Shape analysis
    cluster_shape_analysis(pos[:,:,frames], labels, FData.config_['box_size'],
            datapath=params['data_filestream'])
    cluster_shape_analysis_extent(pos[:,:,frames], labels, FData.config_['box_size'],
            save=True, savepath=params['plot_path'] / 'cluster_filament_extent.pdf',
            datapath=params['data_filestream'])
    
    # Nematic order
    cluster_nematic_order(pos[:,:,frames], FData.orientation_[:,:,frames],labels,
            datapath=params['data_filestream'])

def condensed_via_xlinks_and_mobility(XData, FData, savepath=None, save=False, datapath=None):
    """ cluster using num xlinks and filament mobility """

    # Get Data
    # Crosslinker number
    nxs,nxd = calc_num_xlink_filament(XData.link0_, XData.link1_, FData.nfil_)
    nx = nxd

    # Mobility
    pos_unfolded = FData.unfold_trajectories('com')
    mobi = calc_mobility(pos_unfolded)

    # Convert to dataframe
    names = ['Crosslinker Number', 'Mobility']
    data = np.zeros((nx.size,2))
    data[:,0] = nx.flatten()
    data[:,1] = mobi.flatten()
    # df = pd.DataFrame(data, columns=names)

    # Clustering: kmeans 
    n_clusters = 2
    X_std = StandardScaler().fit_transform(data)
    # print('KMeans clustering: N_clusters = {}'.format(n_clusters))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_std)
    labels = kmeans.labels_
    
    # Edit labels so that free filaments are label -1, and condensed ones are 0
    if kmeans.cluster_centers_[0,0] > kmeans.cluster_centers_[1,0]:
        labels_cpy = labels.copy()
        labels_cpy[ labels == 1] = -1 
        labels = labels_cpy
    else:
        labels_cpy = labels.copy()
        labels_cpy[ labels == 0] = -1 
        labels_cpy[ labels == 1] = 0
        labels = labels_cpy

    labels = np.reshape(labels, nx.shape)

    # Plot filaments of last frame
    if save:
        pos = FData.get_com()[:,:,-1]
        fig,ax = plt.subplots(1,3, figsize=(12,4))
        colors = ['Gray','Teal']
        alphas = [0.1,0.3]
        labs= ['Vapor', 'Condensed']
        for jc in range(2):
            # plot all points
            idx = labels[:,-1]==jc-1
            ax[0].scatter( pos[0,idx], pos[1,idx], color=colors[jc], label=labs[jc], s=2, alpha=alphas[jc])
            ax[1].scatter( pos[0,idx], pos[2,idx], color=colors[jc], label=labs[jc], s=2, alpha=alphas[jc])
            ax[2].scatter( pos[1,idx], pos[2,idx], color=colors[jc], label=labs[jc], s=2, alpha=alphas[jc])
        ax[0].legend()
        ax[0].set(xlabel=r'X ($\mu m$)',ylabel=r'Y ($\mu m$)')
        ax[1].set(xlabel=r'X ($\mu m$)',ylabel=r'Z ($\mu m$)')
        ax[2].set(xlabel=r'Y ($\mu m$)',ylabel=r'Z ($\mu m$)')
        plt.tight_layout()
        plt.savefig(savepath)
    plt.close()

    ratio = np.zeros( labels.shape[1])
    for jt in range(labels.shape[1]):
        ratio[jt] = np.sum(labels[:,jt] != -1) / labels.shape[0]

    print('Condensed Fraction = {0:.2f} +- {1:.3f}'.format(
        np.mean(ratio[-50:]), np.std(ratio[-50:])))
    
    if datapath is not None:
        # save ratio to h5py
        datapath.create_dataset('filament/condensed_ratio', data=ratio, dtype='f')

    return labels, ratio

def PlotTimeAvgLabelHist(labels, savepath):
    fig,ax = plt.subplots()
    ax.hist(np.mean(labels, axis=1), bins=50)
    ax.set(yscale='log', title='Time Averaged Label', xlabel='Mean Label', ylabel='Count')
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

def PlotRatioCondensedFilaments(ratio, savepath):

    fig,ax = plt.subplots()
    ax.plot(ratio, color='blue', lw=2)
    ax.set(xlabel='Frame')
    ax.set(ylabel='Ratio of Condensed Fil')
    ax.set_ylim(bottom=-0.02, top=1.02)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def cluster_via_dbscan(pos_all, box_size, savepath=None, save=False):
    """ cluster using dbscan """

    min_biggest_cluster_size = 10 
    min_smallest_cluster_ratio = 0.2 
    eps = 0.05 # min neighbor distance for identification of cores
    min_samples = 6 # min size of core

    # frames
    frames = np.arange(pos_all.shape[2])

    # Labels
    labels_all = np.zeros((pos_all.shape[1], pos_all.shape[2]))

    for jframe in frames:

        # Clustering Algorithm DBSCAN
        pos = pos_all[:,:,jframe]
        X = pos[:,:].transpose()

        # Get distances PBC
        X = pdist_pbc(X,box_size)

        # Compute DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(X)
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        size_cluster = [np.sum(labels==ii) for ii in range(n_clusters_)]
        # Only keep clusters with min_size
        for jc in range(n_clusters_):
            if size_cluster[jc] < min_biggest_cluster_size:
                labels[labels==jc] = -1 # assign as noise
        # Toss clusters that are too small compared to biggest cluster
        min_size = min_smallest_cluster_ratio*np.max(size_cluster)
        for jc in range(n_clusters_):
            if size_cluster[jc] < min_size:
                labels[labels==jc] = -1 # assign as noise

        # Change cluster labels
        for idx,c_label in enumerate(set(labels)):
            if c_label != -1:
                labels[ labels==c_label] = idx
            
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        # if n_clusters_ > 1:
            # pdb.set_trace()
            # print('whoops')
        size_cluster = [np.sum(labels==ii) for ii in range(n_clusters_)]
        n_noise_ = list(labels).count(-1)
        # print("Estimated number of clusters: %d" % n_clusters_)
        # print("Estimated number of noise points: %d" % n_noise_)

        labels_all[:,jframe] = labels
        
        if save:
            fig,ax = plt.subplots(1,3, figsize=(12,4))
            colors = sns.color_palette("husl", n_clusters_)
            for jc in range(n_clusters_):
                # plot all points
                ax[0].scatter( pos[0,labels==jc], pos[1,labels==jc], color=colors[jc], label=jc, s=2, alpha=0.3)
                ax[1].scatter( pos[0,labels==jc], pos[2,labels==jc], color=colors[jc], label=jc, s=2, alpha=0.3)
                ax[2].scatter( pos[1,labels==jc], pos[2,labels==jc], color=colors[jc], label=jc, s=2, alpha=0.3)
                # plot com
                com = np.mean( pos[:,labels==jc], axis=1)
                ax[0].scatter( [com[0]],[com[1]],color=colors[jc],marker='x',s=20)
                ax[1].scatter( [com[0]],[com[2]],color=colors[jc],marker='x',s=20)
                ax[2].scatter( [com[1]],[com[2]],color=colors[jc],marker='x',s=20)
            ax[0].legend()
            ax[0].set(xlabel=r'X ($\mu m$)',ylabel=r'Y ($\mu m$)')
            ax[1].set(xlabel=r'X ($\mu m$)',ylabel=r'Z ($\mu m$)')
            ax[2].set(xlabel=r'Y ($\mu m$)',ylabel=r'Z ($\mu m$)')
            plt.tight_layout()
            plt.savefig(savepath)
            save=False

    plt.close()
    return labels_all

def cluster_shape_analysis_extent(pos, labels, box_size, savepath=None, save=False, datapath=None):
    """ cluster shape analysis extent """

    # frames
    frames = np.arange(pos.shape[2])

    # Find extent of each cluster
    stdX = []
    stdY = []
    stdZ = []
    for jframe in frames:
        n_clusters_ = len(set(labels[:,jframe])) - (1 if -1 in labels else 0)
        for jc in range(n_clusters_):
            cpos = pos[:,labels[:,jframe]==jc, jframe]
            xyz_max = pdist_pbc_xyz_max(cpos.transpose(), box_size)
            stdX.append( xyz_max[0])
            stdY.append( xyz_max[1])
            stdZ.append( xyz_max[2])
    std = np.zeros((3,len(stdX)))
    std[0,:] = stdX
    std[1,:] = stdY
    std[2,:] = stdZ
    dims = ['X','Y','Z']
    for jd in range(len(dims)):
        print('Extent in dim {0} = {1:.3f} {2} +- {3:.3f} {2}'.format( dims[jd],
            np.mean(std[jd,:]), r'$\mu m$', np.std(std[jd,:]) ))
    
    if save:
        colors = sns.color_palette("husl", 3)
        fig,ax = plt.subplots()
        bins = np.linspace(0,np.max(std.flatten()),13)
        for jd in range(3):
            ax.hist( std[jd,:], bins, color=colors[jd], label=dims[jd], alpha=0.4)
        ax.legend()
        ax.set(xlabel=r'XYZ ($\mu m$)',ylabel='Count')
        plt.tight_layout()
        plt.savefig(savepath)
    plt.close()
    
    if datapath is not None:
        # save ratio to h5py
        datapath.create_dataset('filament/cluster_extent_xyz', data=std, dtype='f')

def cluster_shape_analysis(pos, labels, box_size, datapath=None):
    """ cluster shape analysis """

    # frames
    frames = np.arange(pos.shape[2])

    # Find shape parameters
    rg2 = []
    asphericity = []
    acylindricity = []
    anisotropy = []
    for jframe in frames:
        n_clusters_ = len(set(labels[:,jframe])) - (1 if -1 in labels else 0)
        for jc in range(n_clusters_):
            cpos = pos[:,labels[:,jframe]==jc, jframe]
            _,G_data = calc_gyration_tensor3d_pbc(cpos.transpose(), box_size)
            asphericity.append( G_data["asphericity"])
            acylindricity.append( G_data["acylindricity"])
            anisotropy.append( G_data["shape_anisotropy"])
            rg2.append( G_data["Rg2"])
    
    if datapath is not None:
        datapath.create_dataset('filament/asphericity', data=asphericity, dtype='f')
        datapath.create_dataset('filament/acylindricity', data=acylindricity, dtype='f')
        datapath.create_dataset('filament/anisotropy', data=anisotropy, dtype='f')
        datapath.create_dataset('filament/rg2', data=rg2, dtype='f')

def cluster_nematic_order(pos, ort, labels, datapath=None):
    
    # frames
    frames = np.arange(pos.shape[2])

    # Find max eigenvalue for each cluster
    Smax = []
    Sxyz = []
    for jframe in frames:
        n_clusters_ = len(set(labels[:,jframe])) - (1 if -1 in labels else 0)
        for jc in range(n_clusters_):
            idx = labels[:,jframe]==jc
            
            # orientations in cluster
            ort0 = ort[:,idx,jframe]

            # Max eigenvalue
            Q = calc_nematic_tensor(ort0)
            Smax.append( np.sqrt(np.tensordot(Q, Q)*1.5) )
            
            # Sx,Sy,Sz
            cos_angle_sq = ort0**2
            Sxyz.append( 0.5*np.mean( 3*cos_angle_sq - 1, axis=1))
    Sxyz = np.array(Sxyz).transpose()
    Smax = np.array(Smax)

    print('Scalar Order Parameter = {0:.3f} +- {1:.3f}'.format( np.mean(Smax), np.std(Smax) ))
    print('Sx = {0:.3f} +- {1:.3f}'.format( np.mean(Sxyz[0,:]), np.std(Sxyz[0,:]) ))
    print('Sy = {0:.3f} +- {1:.3f}'.format( np.mean(Sxyz[1,:]), np.std(Sxyz[1,:]) ))
    print('Sz = {0:.3f} +- {1:.3f}'.format( np.mean(Sxyz[2,:]), np.std(Sxyz[2,:]) ))

    if datapath is not None:
        # save ratio to h5py
        datapath.create_dataset('filament/cluster_Smax', data=Smax, dtype='f')
        datapath.create_dataset('filament/cluster_Sxyz', data=Sxyz, dtype='f')

def condensed_msd_diffusion(pos, labels, N=100, dt=1, save=False, savepath=None, datapath=None):
    
    # Get the filaments that are condensed for the last N frames
    idx = np.sum( labels[:,-1*N:], axis=1) == 0
    pos_condensed = pos[:,idx,-1*N:]

    # MSD
    timeArray = dt * np.arange(N)
    MSD = calc_msd_fft( pos_condensed)
    slopes_mu = np.diff( np.mean(MSD,axis=0) ) / np.diff( timeArray)
    slopes_sem = np.diff( np.std(MSD,axis=0)/np.sqrt(MSD.shape[0]) ) / np.diff( timeArray)
    slope_middle = np.array([ slopes_mu[int( len(slopes_mu)/2 ) ], slopes_sem[int( len(slopes_sem)/2 ) ] ])
    print('MSD slope (center) = {0:.5f} +- {1:.5f} {2}'.format(slope_middle[0], slope_middle[1], r'$\mu m^2 s^{-1}$' ))

    # Plotting
    fig,ax = plt.subplots()
    ax.plot(timeArray, np.mean(MSD, axis=0), color='blue', lw=2, label='N = {0}'.format(MSD.shape[0]) )
    ax.fill_between(timeArray,
                    np.mean(MSD, axis=0) - np.std(MSD, axis=0)/np.sqrt(MSD.shape[0]),
                    np.mean(MSD, axis=0) + np.std(MSD, axis=0)/np.sqrt(MSD.shape[0]),
                    color='blue', alpha=0.2)
    ax.set_xlim(left=-0.01)
    ax.set_ylim(bottom=-0.01)
    ax.legend()
    ax.set(xlabel='Lag time / s')
    ax.set(ylabel=r'MSD / $\mu m^2$')
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

    if datapath is not None:
        # save ratio to h5py
        datapath.create_dataset('filament/condensed_msd_slope', data=slope_middle, dtype='f')

def pdist_pbc(pos, box_size):
    """ pos is dimension N x 3 """
    for jd in range(pos.shape[1]):
        pd = pdist(pos[:,jd].reshape(-1,1))
        pd[ pd > 0.5*box_size[jd] ] -= box_size[jd]
        try:
            total += pd**2
        except:
            total = pd**2
    total = squareform( np.sqrt(total) )
    return total

def pdist_pbc_xyz_max(pos, box_size):
    """ pos is dimension N x 3 """
    dist_xyz = np.zeros(3)
    for jd in range(pos.shape[1]):
        pd = pdist(pos[:,jd].reshape(-1,1))
        pd[ pd > 0.5*box_size[jd] ] -= box_size[jd]
        dist_xyz[jd] = np.max( np.abs(pd) )
    return dist_xyz 

def calc_com_pbc(pos, box_size):
    """ calulate com of pos (N x 3) with pbc """
    
    # Map each dimension from a line to a circle.
    com = np.zeros(pos.shape[1])
    for jd in range(pos.shape[1]):
        angle = 2*np.pi*pos[:,jd]/box_size[jd]

        # Get coordinates for a point on a circle for each particle in this dimension
        costheta = np.cos(angle)
        sintheta = np.sin(angle)

        # Get mean coordinate in this dimension
        costheta_com = np.mean(costheta)
        sintheta_com = np.mean(sintheta)

        # Find an angle associated with this point
        angle_com = np.arctan2(-1*sintheta_com, -1*costheta_com) + np.pi
        
        # Transform back to line
        com[jd] = angle_com*box_size[jd]/(2*np.pi)

    return com

def packing_fraction( pos_minus, pos_plus, config, labels, 
        save=False, savepath=None, datapath=None):

    # Condensed
    idx = labels[:,-1] == 0
    pf_c = calc_local_packing_fraction_frame(
            pos_minus[:,idx], pos_plus[:,idx], config['diameter_fil'],
            config["box_size"])

    # Vapor
    idx = labels[:,-1] == -1
    pf_v = calc_local_packing_fraction_frame(
            pos_minus[:,idx], pos_plus[:,idx], config['diameter_fil'],
            config["box_size"])

    # check
    measured_total_pf = ( np.mean(pf_c)*np.sum(labels[:,-1]==0) + 
            np.mean(pf_v)*np.sum(labels[:,-1]==-1) )/labels[:,-1].shape[0]
    print('Measured Packing Fraction = {0:.3f}'.format(measured_total_pf))
    
    # Plotting
    if save:
        fig,ax = plt.subplots()
        colors = sns.color_palette("husl", 2)
        bins = np.linspace(0,np.max([np.max(pf_c), np.max(pf_v)]),12)
        ax.hist( pf_c, bins, color=colors[0], label='Condensed', alpha=0.6)
        ax.hist( pf_v, bins, color=colors[1], label='Vapor', alpha=0.6)
        ax.legend()
        ax.set(xlabel='Packing Fraction',ylabel='Count')
        plt.tight_layout()
        plt.savefig(savepath)

    if datapath is not None:
        # save ratio to h5py
        datapath.create_dataset('filament/condensed_pf', data=pf_c, dtype='f')
        datapath.create_dataset('filament/vapor_pf', data=pf_v, dtype='f')

            

