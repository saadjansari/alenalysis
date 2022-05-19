import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
import seaborn as sns
from src.calc_gyration_tensor import calc_gyration_tensor3d_pbc
from src.calc_com_pbc import calc_com_pbc
from src.calc_pdist_pbc import pdist_pbc, pdist_pbc_xyz_max

# PlotXlinkClusters {{{
def PlotXlinkClusters(XData, params, N=100):
    """ Plot the xlinker clusters"""
    
    # Get last N frames
    frames = np.arange(XData.nframe_-N,XData.nframe_)

    # Crosslinker clusters
    print('Crosslinker clusters...') 
    pos_com = XData.get_com()
    labels = cluster_via_dbscan(pos_com[:,:,frames], XData.config_['box_size'],
            save=True, savepath=params['plot_path'] / 'cluster_xlinker_dbscan.pdf')

    # Plot radial distance from cluter COM
    cluster_radial_distances(XData.pos_minus_[:,:,frames], XData.pos_plus_[:,:,frames], labels, XData.config_['box_size'],
            save=True, savepath=params['plot_path'] / 'cluster_xlinker_radial_distribution.pdf',
            datapath=params['data_filestream'])

    # Shape analysis
    cluster_shape_analysis(XData.pos_minus_[:,:,frames], XData.pos_plus_[:,:,frames], labels, XData.config_['box_size'],
            datapath=params['data_filestream'])
    # cluster_shape_analysis_extent(pos_com[:,:,frames], labels, XData.config_['box_size'],
            # save=True, savepath=params['plot_path'] / 'cluster_xlinker_extent.pdf',
            # datapath=params['data_filestream'])

# }}}

# cluster_via_dbscan {{{
def cluster_via_dbscan(pos_all, box_size, savepath=None, save=False):
    """ cluster using dbscan """

    min_cluster_size = 20
    eps = 0.05 # min neighbor distance for identification of cores
    min_samples = 10 # min size of core

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
        # from sklearn.neighbors import NearestNeighbors
        # neighbors = NearestNeighbors(n_neighbors=min_samples)
        # neighbors_fit = neighbors.fit(pos.transpose())
        # distances, indices = neighbors_fit.kneighbors(pos.transpose())
        # distances = np.sort(distances, axis=0)[:,1]
        # plt.plot(distances);plt.show()
        # pdb.set_trace()
        # plt.close()
        # print('whoops')
        # size_cluster = [np.sum(labels==ii) for ii in range(n_clusters_)]
        # n_noise_ = list(labels).count(-1)
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
                com = calc_com_pbc( pos[:,labels==jc].transpose(), box_size)
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
# }}}

# cluster_shape_analysis_extent {{{
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
        if 'xlinker/cluster_extent_xyz' in datapath:
            del datapath['xlinker/cluster_extent_xyz']
        datapath.create_dataset('xlinker/cluster_extent_xyz', data=std, dtype='f')
    return labels
# }}}

# cluster_shape_analysis {{{
def cluster_shape_analysis(pos0, pos1, labels, box_size, datapath=None):
    """ cluster shape analysis """

    # frames
    frames = np.arange(pos0.shape[2])

    # Find shape parameters
    rg2 = []
    asphericity = []
    acylindricity = []
    anisotropy = []
    for jframe in frames:
        n_clusters_ = len(set(labels[:,jframe])) - (1 if -1 in labels else 0)
        for jc in range(n_clusters_):
            p0 = pos0[:,labels[:,jframe]==jc, jframe]
            p1 = pos1[:,labels[:,jframe]==jc, jframe]
            pos_sampled = np.linspace(p0, p1, 5, axis=1).reshape((p0.shape[0], -1), order='F')
            _,G_data = calc_gyration_tensor3d_pbc(pos_sampled.transpose(), box_size)
            asphericity.append( G_data["asphericity"])
            acylindricity.append( G_data["acylindricity"])
            anisotropy.append( G_data["shape_anisotropy"])
            rg2.append( G_data["Rg2"])
    
    if datapath is not None:
        if 'xlinker/asphericity' in datapath:
            del datapath['xlinker/asphericity']
        if 'xlinker/acylindricity' in datapath:
            del datapath['xlinker/acylindricity']
        if 'xlinker/anisotropy' in datapath:
            del datapath['xlinker/anisotropy']
        if 'xlinker/rg2' in datapath:
            del datapath['xlinker/rg2']
        datapath.create_dataset('xlinker/asphericity', data=asphericity, dtype='f')
        datapath.create_dataset('xlinker/acylindricity', data=acylindricity, dtype='f')
        datapath.create_dataset('xlinker/anisotropy', data=anisotropy, dtype='f')
        datapath.create_dataset('xlinker/rg2', data=rg2, dtype='f')
# }}}

# cluster_radial_distances {{{
def cluster_radial_distances(pos0, pos1, labels, box_size, save=False, savepath=None, datapath=None):
    """ cluster radial distances"""

    # frames
    frames = np.arange(pos0.shape[2])

    # com
    pos_com = (pos0+pos1)/2
    
    # bins
    bins = np.linspace(0,0.5,201)

    DD = np.zeros( (len(bins)-1, len(frames)))

    # For each cluster, compute distances from c.o.m of cluster
    for jframe in frames:
        n_clusters_ = len(set(labels[:,jframe])) - (1 if -1 in labels else 0)
        for jc in range(n_clusters_):
            p0 = pos0[:,labels[:,jframe]==jc, jframe]
            p1 = pos1[:,labels[:,jframe]==jc, jframe]
            pos_sampled = np.linspace(p0, p1, 5, axis=1).reshape((p0.shape[0], -1), order='F')
            com = calc_com_pbc( pos_com[:,labels[:,jframe]==jc, jframe].transpose(), box_size)
            
            # Distance from C.O.M
            d_xyz = pos_sampled-com.reshape(-1,1)
            for jdim in range(len(com)):
                d_xyz[jdim,:][ d_xyz[jdim,:] < -0.5*box_size[jdim] ] += box_size[jdim]
                d_xyz[jdim,:][ d_xyz[jdim,:] > 0.5*box_size[jdim] ] -= box_size[jdim]
            distances = np.linalg.norm(d_xyz, axis=0)
            DD[:,jframe] += np.histogram(distances,bins)[0]
        DD[:,jframe] = DD[:,jframe] / np.sum( DD[:,jframe]*np.diff(bins))

    if save and savepath is not None:
        fig,(ax,ax2) = plt.subplots(1,2,figsize=(8,3))
        im = ax.imshow(DD.T, cmap='viridis', interpolation='gaussian', 
                aspect='auto', vmin=0, 
                extent=[bins[0],bins[-1],0,frames[-1]])
        ax.set(ylabel='Frame', xlabel=r'Distance from cluster COM ($\mu m$)')
        ax.set_xticks(np.arange(0,bins[-1]+0.0001,0.1))
        plt.colorbar(im, ax=ax, label='Probability density')
        
        ax2.plot( 0.5*(bins[1:]+bins[:-1]), np.mean(DD,axis=1), color='k',label='PDF', linewidth=2)
        rmax_idx = np.where( np.mean(DD,axis=1) == np.max( np.mean(DD,axis=1) ))[0][0]
        rmax = np.around(np.mean(bins[rmax_idx:rmax_idx+2]),4)
        ax2.axvline( rmax, color='k', linestyle='dotted', label=r'Peak = {0:.4f} $\mu m$'.format(rmax) )
        ax2.set(ylabel='Probability density', xlabel=r'Distance from cluster COM ($\mu m$)')
        ax2.set_xticks(np.arange(0,bins[-1]+0.0001,0.1))
        ax2.set_ylim(top=ax2.get_ylim()[1]*1.2)
        ax2.legend(loc='upper right', frameon=False)

        plt.tight_layout()
        plt.savefig(savepath, bbox_inches="tight")
        plt.close()
        
    if datapath is not None:
        if 'xlinker/com_dist_val' in datapath:
            del datapath['xlinker/com_dist_val']
        if 'xlinker/com_dist_pdf' in datapath:
            del datapath['xlinker/com_dist_pdf']
        datapath.create_dataset('xlinker/com_dist_val', data=0.5*(bins[1:]+bins[:-1]), dtype='f')
        datapath.create_dataset('xlinker/com_dist_pdf', data=np.mean(DD,axis=1), dtype='f')
# }}}
