from numba import njit
import numpy as np
from matplotlib import pyplot as plt
import decorators
import pdb
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from CalcCondensationFilaments import pdist_pbc, pdist_pbc_xyz_max

def PlotXlinkClusters(XData, params, N=10):
    """ Plot the xlinker clusters"""
    
    # Get last N frames
    frames = np.arange(XData.nframe_-N,XData.nframe_)

    # Crosslinker clusters
    print('Crosslinker clusters...') 
    pos_com = XData.get_com()
    labels = cluster_via_dbscan(pos_com[:,:,frames], XData.config_['box_size'],
            save=True, savepath=params['plot_path'] / 'cluster_xlinker_dbscan.pdf')

    # Shape analysis
    cluster_shape_analysis_extent(pos_com[:,:,frames], labels, XData.config_['box_size'],
            save=True, savepath=params['plot_path'] / 'cluster_xlinker_extent.pdf',
            datapath=params['data_filestream'])


def cluster_via_dbscan(pos_all, box_size, savepath=None, save=False):
    """ cluster using dbscan """

    min_cluster_size = 50
    eps = 0.2 # min neighbor distance for identification of cores
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
            if size_cluster[jc] < min_cluster_size:
                labels[labels==jc] = -1 # assign as noise

        # Change cluster labels
        for idx,c_label in enumerate(set(labels)):
            if c_label != -1:
                labels[ labels==c_label] = idx
            
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        size_cluster = [np.sum(labels==ii) for ii in range(n_clusters_)]
        n_noise_ = list(labels).count(-1)
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)

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
        for jd in range(3):
            ax.hist( std[jd,:], 12, color=colors[jd], label=dims[jd], alpha=0.7)
        ax.legend()
        ax.set(xlabel=r'XYZ ($\mu m$)',ylabel='Count')
        plt.tight_layout()
        plt.savefig(savepath)
    plt.close()
    if datapath is not None:
        # save ratio to h5py
        datapath.create_dataset('xlinker/cluster_extent_xyz', data=std, dtype='f')
    return labels
