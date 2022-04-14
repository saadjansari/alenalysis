from numba import njit
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import src.decorators
from src.CalcMobility import calc_mobility
from src.CalcNumXlinks import calc_num_xlink_filament
import pdb
from pathlib import Path
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import seaborn as sns
from src.CalcOrderParameters import calc_nematic_tensor, calc_nematic_basis
from src.CalcMSD import calc_msd_fft
from scipy.spatial.distance import pdist, squareform
from src.calc_gyration_tensor import calc_gyration_tensor3d_pbc
from src.CalcPackingFraction import calc_local_packing_fraction_frame
from src.write2vtk import add_array_to_vtk
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from src.unfold_trajectories import unfold_trajectories_njit, unfold_trajectory_njit
from src.DataHandler import AddCorrelationsToDataSeries
from src.multicolored_line import plot_multicolored_lines
from src.CalcBundleTwist import calc_bundle_twist
from src.calc_com_pbc import calc_com_pbc
from src.calc_pdist_pbc import pdist_pbc, pdist_pbc_xyz_max
from src.connected_components import get_nodes_in_clusters

# PlotFilamentCondensation {{{
def PlotFilamentCondensation(FData, XData, params, write2vtk=False):
    """ Plot the filament condensation """

    plot_ratio_condensed_filaments=True
    plot_diffusion=True
    plot_packing_fraction=False
    plot_local_order=False
    plot_time_averaged_label=False
    plot_trajectories=True
    plot_residence_times = False
    plot_bundle_twist = True
    plot_aspect_ratio= True
    
    print('Filaments condensation...') 
    # FData = AddCorrelationsToDataSeries(FData, params['sim_path'])
    # labels, ratio = condensed_via_xlinks_and_mobility(XData, FData, save=True, 
            # savepath=params['plot_path'] / 'condensed_filament_positions.pdf',
            # datapath=params['data_filestream'], write2vtk=write2vtk, simpath=params['sim_path'])
    # labels, ratio = condensed_multidimensional(XData, FData, save=True, 
            # savepath=params['plot_path'] / 'condensed_filament_positions.pdf',
            # datapath=params['data_filestream'], write2vtk=write2vtk, simpath=params['sim_path'])
    labels, ratio = condensed_networkx(XData, FData, params, save=True, 
            savepath=params['plot_path'] / 'condensed_filament_positions.pdf',
            datapath=params['data_filestream'], write2vtk=write2vtk, simpath=params['sim_path'])

    if plot_aspect_ratio:
        condensed_aspect_ratio(FData.get_com(), FData.orientation_,labels, 
                FData.config_['box_size'],
                save=True, savepath=params['plot_path'] / 'condensed_aspect_ratio.pdf',
                datapath=params['data_filestream'])

    if plot_bundle_twist:
        twist,rdist = CalcBundleTwist(FData, labels, N=100,
                savepath=params['plot_path'] / 'condensed_bundle_twist.pdf',
                datapath=params['data_filestream'])
        PlotHistBundleTwist_vs_RadialDist(twist,rdist,
                savepath=params['plot_path'] / 'condensed_bundle_twist_vs_radial_dist.pdf')
        Plot2DBundleTwist_NematicBasis(FData, labels,
                savepath=params['plot_path'] / 'condensed_bundle_twist_cartesian.pdf')
        Plot3DBundleTwist(FData, labels, nematic_basis=False,
                savepath=params['plot_path'] / 'condensed_bundle_twist_3d.pdf')
        Plot3DBundleTwist(FData, labels, nematic_basis=True,
                savepath=params['plot_path'] / 'condensed_bundle_twist_3d_nematic.pdf')

    if plot_trajectories:
        PlotCondensedTrajectories(FData, labels, N=100,
            savepath=params['plot_path'] / 'condensed_trajectories.pdf')

    # Plot ratio of condensed filaments over time
    if plot_ratio_condensed_filaments:
        PlotRatioCondensedFilaments(ratio, FData.nfil_, params['plot_path'] / 'condensed_filament_ratio.pdf')

    # Condensed MSD
    if plot_diffusion:
        pos_unfolded = FData.unfold_trajectories('com')
        condensed_msd_diffusion(pos_unfolded, labels, save=True, dt=0.05,
                savepath=params['plot_path'] / 'condensed_filament_msd.pdf',
                datapath=params['data_filestream'])

    # Condensed/Vapor PF
    if plot_packing_fraction:
        packing_fraction( 
                FData.pos_minus_[:,:,-1],
                FData.pos_plus_[:,:,-1],
                FData.config_,
                labels,
                save=True,
                savepath=params['plot_path'] / 'filaments_pf.pdf',
                datapath=params['data_filestream'])
            
    # Plot time averaged label
    if plot_time_averaged_label:
        PlotTimeAvgLabelHist(labels, params['plot_path'] / 'condensed_filament_time_avg_label.pdf')

    # Plot Local Polar Order
    if plot_local_order:
        FData = AddCorrelationsToDataSeries(FData, params['sim_path'])
        PlotCondensedLocalOrder(FData, labels,
                savedir=params['plot_path'],
                datapath=params['data_filestream'])

    if plot_residence_times:
        PlotResidenceTimes(labels, N=800, dt=0.05,
                savepath=params['plot_path'] / 'condensed_residence_times.pdf',
                datapath=params['data_filestream'])
        
# }}}

# PlotFilamentClusters {{{
def PlotFilamentClusters(FData, params, N=50):
    """ Plot the filament plus-end clusters """
    
    # Get last N frames
    frames = np.arange(FData.nframe_-N,FData.nframe_)

    # Filament plus-end clusters
    print('Filaments plus-end clusters...') 
    pos = FData.pos_plus_
    labels = cluster_via_dbscan(pos[:,:,frames], FData.config_['box_size'],
            save=True, savepath=params['plot_path'] / 'cluster_filament_dbscan.pdf')

    # Track labels
    labels_tracked = TrackClusterLabels(labels)

    # PLot Tracked Clusters
    PlotClusterMotion(pos[:,:,frames], labels_tracked, FData.config_['box_size'],
            savepath=params['plot_path'] / 'cluster_tracked_motion.pdf')

    # Plot size distirbution histogram
    PlotHistClusterNumbers(labels, savepath=params['plot_path']/'cluster_filament_numbers.pdf', save=True, datapath=params['data_filestream'])

    # Plot cluster astral order
    # PlotClusterAstralOrder( FData.pos_plus_[:,:,frames], FData.pos_minus_[:,:,frames], 
            # labels, FData.config_['box_size'], 
            # savepath=params['plot_path']/'cluster_filament_astral_order.pdf', 
            # save=True, datapath=params['data_filestream'])
            
    # Plot cluster astral order 2D
    PlotClusterAstralOrder( FData.pos_plus_[:2,:,frames], FData.pos_minus_[:2,:,frames], 
            labels, FData.config_['box_size'][:2], 
            savepath=params['plot_path']/'cluster_filament_astral_order_XY.pdf', 
            save=True, datapath=params['data_filestream'])

    # Filament diffusion within cluster
    cluster_msd_diffusion(FData, labels_tracked, save=True, dt=0.05,
            savepath=params['plot_path'] / 'cluster_filament_msd.pdf',
            datapath=params['data_filestream'])

    # Shape analysis
    cluster_shape_analysis(pos[:,:,frames], labels, FData.config_['box_size'],
            datapath=params['data_filestream'])
    cluster_shape_analysis_extent(pos[:,:,frames], labels, FData.config_['box_size'],
            save=True, savepath=params['plot_path'] / 'cluster_filament_extent.pdf',
            datapath=params['data_filestream'])
    
    # Nematic order
    cluster_nematic_order(pos[:,:,frames], FData.orientation_[:,:,frames],labels,
            datapath=params['data_filestream'])
# }}}

# condensed_via_xlinks_and_mobility {{{
def condensed_via_xlinks_and_mobility(XData, FData, savepath=None, save=False, datapath=None, write2vtk=False, simpath=None):
    """ cluster using num xlinks and filament mobility """

    # Get Data
    # Use last N frames
    N=200
    # Crosslinker number
    nxs,nxd = calc_num_xlink_filament(XData.link0_, XData.link1_, FData.nfil_)
    nx = nxd
    # Set any nx<3 -----> 0
    nx[ nx < 3] = 0
    nx_use = nx[:,-1*N:]

    # Mobility
    pos_unfolded = FData.unfold_trajectories('plus')
    mobi = calc_mobility(pos_unfolded, windowSize=5)
    mobi_use = mobi[:,-1*N:]

    # Convert to dataframe
    names = ['Crosslinker Number', 'Mobility']
    data_fit = np.zeros((nx_use.size,2))
    data_fit[:,0] = nx_use.flatten()
    data_fit[:,1] = mobi_use.flatten()
    data = np.zeros((nx.size,2))
    data[:,0] = nx.flatten()
    data[:,1] = mobi.flatten()

    n_clusters = 2
    scaler = StandardScaler().fit(data)
    X_std = scaler.transform(data)

    model = GaussianMixture(n_components=n_clusters).fit( data_fit)
    centers = model.means_
    # Model score
    print('GMM BIC score = {0:.2f}'.format( model.bic(data) ) )

    # labels = model.predict( scaler.transform(data))
    labels = model.predict( data)
    if model.bic(data) < 0:
        print('Clustering score too low! No condensed state found')
        labels[:] = -1
        labels = np.reshape(labels, nx.shape)

    else:
        
        # Edit labels so that free filaments are label -1, and condensed ones are 0
        clabel = np.argsort(centers[:,0])[-1]

        labels[labels!=clabel] = -1
        labels[labels==clabel] = 0 

        # set any uncertain mpoints to vapor
        labels[ np.where(np.max(model.predict_proba( data), axis=1)<0.8)[0] ] = -1
        labels = np.reshape(labels, nx.shape)

    # Plot filaments of last frame
    if save:
        jf = -1
        p0 = FData.pos_minus_[:,:,jf]
        p1 = FData.pos_plus_[:,:,jf]
        fig,ax = plt.subplots(1,3, figsize=(12,4))
        colors = ['Gray','Teal']
        alphas = [0.1,0.3]
        labs= ['Vapor', 'Condensed']
        for jc in range(2):
            # plot all points
            idx = labels[:,-1]==jc-1
            ax[0].scatter( p1[0,idx], p1[1,idx], color=colors[jc], label=labs[jc], s=2, alpha=alphas[jc])
            ax[1].scatter( p1[0,idx], p1[2,idx], color=colors[jc], label=labs[jc], s=2, alpha=alphas[jc])
            ax[2].scatter( p1[1,idx], p1[2,idx], color=colors[jc], label=labs[jc], s=2, alpha=alphas[jc])
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
        if 'filament/condensed_ratio' in datapath:
            del datapath['filament/']
        datapath.create_dataset('filament/condensed_ratio', data=ratio, dtype='f')
        
    if write2vtk:
        add_array_to_vtk(labels, 'Condensed', simpath)

    return labels, ratio
# }}}

# condensed_multidimensional {{{
def condensed_multidimensional(XData, FData, savepath=None, save=False, datapath=None, write2vtk=False, simpath=None):
    """ cluster using multidimensional"""

    # Get Data
    # Use last N frames
    nframe = FData.nframe_
    N=200
    
    # Local nematic order 
    lpo = FData.local_nematic_order
    # SET ANY NANS TO 0
    lpo[ np.isnan(lpo)] = 0
    nframe = min([nframe, lpo.shape[-1]])
    lpo = lpo[:,:nframe]
    frames_use = np.arange(nframe-N, nframe)
    lpo_use = lpo[:,frames_use]
    

    # Crosslinker number
    nxs,nxd = calc_num_xlink_filament(XData.link0_, XData.link1_, FData.nfil_)
    nx = nxd[:,:nframe]
    # Set any nx<3 -----> 0
    # nx[ nx < 3] = 0
    nx_use = nx[:,frames_use]

    # mobility
    pos_unfolded = FData.unfold_trajectories('plus')
    mobi = calc_mobility(pos_unfolded, windowSize=5)[:,:nframe]
    mobi_use = mobi[:,frames_use]


    # Convert to dataframe
    names = ['Local Nematic Order', 'Mobility','Crosslinker Number']
    data_fit = np.zeros((nx_use.size,3))
    data_fit[:,0] = lpo_use.flatten()
    data_fit[:,1] = mobi_use.flatten()
    data_fit[:,2] = nx_use.flatten()
    data = np.zeros((nx.size,3))
    data[:,0] = lpo.flatten()
    data[:,1] = mobi.flatten()
    data[:,2] = nx.flatten()

    n_clusters = 2
    scaler = StandardScaler().fit(data)
    X_std = scaler.transform(data)

    model = GaussianMixture(n_components=n_clusters).fit( data_fit)
    centers = model.means_
    # Model score
    print('GMM BIC score = {0:.2f}'.format( model.bic(data) ) )
    print(names)
    print(model.means_) 

    # labels = model.predict( scaler.transform(data))
    # Determine if clustering worked
    # Mean crosslinker number of atleast one cluster above 7
    labels = model.predict( data)
    if np.max(model.means_[:,-1]) < 7:
        print('No crosslinker-dense condensed state found. Clustering result rejected')
        labels[:] = -1
        labels = np.reshape(labels, nx.shape)

    else:
        
        # Edit labels so that free filaments are label -1, and condensed ones are 0
        clabel = np.argsort(centers[:,0])[-1]

        labels[labels!=clabel] = -1
        labels[labels==clabel] = 0 

        # set any uncertain mpoints to vapor
        labels[ np.where(np.max(model.predict_proba( data), axis=1)<0.8)[0] ] = -1
        labels = np.reshape(labels, nx.shape)

    # Plot filaments of last availabel frame
    if save:
        jf = nframe-1
        p0 = FData.pos_minus_[:,:,jf]
        p1 = FData.pos_plus_[:,:,jf]
        fig,ax = plt.subplots(1,3, figsize=(12,4))
        colors = ['Gray','Teal']
        alphas = [0.1,0.3]
        labs= ['Vapor', 'Condensed']
        for jc in range(2):
            # plot all points
            idx = labels[:,-1]==jc-1
            ax[0].scatter( p1[0,idx], p1[1,idx], color=colors[jc], label=labs[jc], s=2, alpha=alphas[jc])
            ax[1].scatter( p1[0,idx], p1[2,idx], color=colors[jc], label=labs[jc], s=2, alpha=alphas[jc])
            ax[2].scatter( p1[1,idx], p1[2,idx], color=colors[jc], label=labs[jc], s=2, alpha=alphas[jc])
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
        if 'filament/condensed_ratio' in datapath:
            del datapath['filament/condensed_ratio']
        datapath.create_dataset('filament/condensed_ratio', data=ratio, dtype='f')
        
    if write2vtk:
        add_array_to_vtk(labels, 'Condensed', simpath)

    labels_full = np.zeros((FData.nfil_,FData.nframe_))
    labels_full[:] = -1
    labels_full[:,:nframe] = labels
    return labels_full, ratio
# }}}

# condensed_networkx{{{
def condensed_networkx(XData, FData, params, savepath=None, save=False, datapath=None, write2vtk=False, simpath=None):
    """ cluster using networkx"""

    nframe = FData.nframe_
    labels = np.zeros( (FData.nfil_, nframe) )
    labels[:] = -1
    
    fastpath_c = params['sim_path'] / 'CondensedLabel.pickle'
    if Path.exists(fastpath_c):
        with open(fastpath_c, "rb") as fp:
            labels_load = pickle.load(fp)
        start_frame = labels_load.shape[1]
        labels[:,:start_frame] = labels_load
    else:
        start_frame = 0
        
    # for each frame, run network analysis to find xlinked cluster
    for jframe in np.arange(start_frame,nframe):
        _,bool_list = get_nodes_in_clusters(
                FData.gid_[:,jframe].astype(int), 
                XData.link0_[:,jframe].astype(int),
                XData.link1_[:,jframe].astype(int),
                )
        labels[bool_list, jframe] = 0
        print('############ Finding Condensed State = {0}/{1} ({2:.0f}%) ############'.format(
            1+jframe,
            nframe, 
            100*(1+jframe)/nframe), end='\r',flush=True)
    print('############ Finding Condensed State = {0}/{0} (100%) ############'.format(nframe))
    with open(fastpath_c, "wb") as fp:
        pickle.dump(labels, fp)

    # Plot filaments of last availabel frame
    if save:
        jf = nframe-1
        p0 = FData.pos_minus_[:,:,jf]
        p1 = FData.pos_plus_[:,:,jf]
        fig,ax = plt.subplots(1,3, figsize=(12,4))
        colors = ['Gray','Teal']
        alphas = [0.1,0.3]
        labs= ['Vapor', 'Condensed']
        for jc in range(2):
            # plot all points
            idx = labels[:,-1]==jc-1
            ax[0].scatter( p1[0,idx], p1[1,idx], color=colors[jc], label=labs[jc], s=2, alpha=alphas[jc])
            ax[1].scatter( p1[0,idx], p1[2,idx], color=colors[jc], label=labs[jc], s=2, alpha=alphas[jc])
            ax[2].scatter( p1[1,idx], p1[2,idx], color=colors[jc], label=labs[jc], s=2, alpha=alphas[jc])
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
        if 'filament/condensed_ratio' in datapath:
            del datapath['filament/condensed_ratio']
        datapath.create_dataset('filament/condensed_ratio', data=ratio, dtype='f')
        
    if write2vtk:
        add_array_to_vtk(labels, 'Condensed', simpath)

    return labels, ratio
# }}}

# PlotTimeAvgLabelHist {{{
def PlotTimeAvgLabelHist(labels, savepath):
    fig,ax = plt.subplots()
    ax.hist(np.mean(labels, axis=1), bins=50)
    ax.set(yscale='log', title='Time Averaged Label', xlabel='Mean Label', ylabel='Count')
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()
# }}}

# PlotRatioCondensedFilaments {{{
def PlotRatioCondensedFilaments(ratio, n_max,savepath):

    fig,ax = plt.subplots()
    ax.plot(ratio, color='k', lw=2)
    ax.set(xlabel='Frame')
    ax.set(ylabel='Ratio of condensed filaments')
    ax.set_ylim(bottom=-0.02, top=1.02)
    ax.axhline(ratio[-1],color='k', linestyle='dotted')
    ax2 = ax.twinx()
    ax2.set(ylabel='Number of condensed filaments')
    ax2.set_ylim(bottom=-0.02*n_max, top=1.02*n_max)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()
# }}}

# cluster_via_dbscan {{{
def cluster_via_dbscan(pos_all, box_size, savepath=None, save=False):
    """ cluster using dbscan """

    min_biggest_cluster_size = 10 
    min_smallest_cluster_ratio = 0.2 
    eps = 0.08 # min neighbor distance for identification of cores
    min_samples = 10 # min size of core

    # frames
    frames = np.arange(pos_all.shape[2])

    # Labels
    labels_all = np.zeros((pos_all.shape[1], pos_all.shape[2]))

    for jframe in frames:

        # Clustering Algorithm DBSCAN
        pos = pos_all[:,:,jframe]
        
        # # Find automated eps
        # neighbors = NearestNeighbors(n_neighbors=min_samples)
        # neighbors_fit = neighbors.fit(pos.transpose())
        # distances, indices = neighbors_fit.kneighbors(pos.transpose())
        # distances = np.sort(distances, axis=0)[:,1]
        # vel = distances[3:]-distances[:-3]
        # idx = 3 + int(len(vel)/2) + np.where(vel[int(len(vel)/2):] > 0.004 )[0][0]
        # eps_auto = distances[idx]
        # eps = eps_auto
        # print('EPS automated = {0}'.format(eps))
        # pdb.set_trace()

        # Get distances PBC
        X = pos[:,:].transpose()
        X = pdist_pbc(X,box_size)

        # Compute DBSCAN
        # eps_all = np.linspace(0.025,0.2,50)
        # min_samples = np.arange(6,20)
        # NCs = np.zeros( (len(min_samples), len(eps_all)))
        # for j1,min_samp in enumerate(min_samples):
            # for j2,eps in enumerate(eps_all):
                # db = DBSCAN(eps=eps, min_samples=min_samp, metric='precomputed').fit(X)
                # labels = db.labels_

                # # Number of clusters in labels, ignoring noise if present.
                # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                # # size_cluster = [np.sum(labels==ii) for ii in range(n_clusters_)]
                # # # Only keep clusters with min_size
                # # for jc in range(n_clusters_):
                    # # if size_cluster[jc] < min_biggest_cluster_size:
                        # # labels[labels==jc] = -1 # assign as noise
                # # # Toss clusters that are too small compared to biggest cluster
                # # min_size = min_smallest_cluster_ratio*np.max(size_cluster)
                # # for jc in range(n_clusters_):
                    # # if size_cluster[jc] < min_size:
                        # # labels[labels==jc] = -1 # assign as noise
                # # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

                # NCs[j1,j2] = n_clusters_

        # pdb.set_trace()

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
        cluster_labels = np.unique( labels)
        cluster_labels = cluster_labels[ cluster_labels != -1]
        for idx,c_label in enumerate( cluster_labels):
            labels[ labels==c_label] = idx
            
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        size_cluster = [np.sum(labels==ii) for ii in range(n_clusters_)]
        n_noise_ = list(labels).count(-1)
        # print("Estimated number of clusters: %d" % n_clusters_)
        # print("Estimated number of noise points: %d" % n_noise_)
        # if n_clusters_ > 2:
            # fig,ax = plt.subplots(1,3, figsize=(12,4))
            # colors = sns.color_palette("husl", n_clusters_)
            # for jc in range(n_clusters_):
                # # plot all points
                # ax[0].scatter( pos[0,labels==jc], pos[1,labels==jc], color=colors[jc], label=jc, s=2, alpha=0.3)
                # ax[1].scatter( pos[0,labels==jc], pos[2,labels==jc], color=colors[jc], label=jc, s=2, alpha=0.3)
                # ax[2].scatter( pos[1,labels==jc], pos[2,labels==jc], color=colors[jc], label=jc, s=2, alpha=0.3)
                # # plot com
                # com = calc_com_pbc( pos[:,labels==jc].transpose(), box_size)
                # ax[0].scatter( [com[0]],[com[1]],color=colors[jc],marker='x',s=20)
                # ax[1].scatter( [com[0]],[com[2]],color=colors[jc],marker='x',s=20)
                # ax[2].scatter( [com[1]],[com[2]],color=colors[jc],marker='x',s=20)
            # ax[0].legend()
            # ax[0].set(xlabel=r'X ($\mu m$)',ylabel=r'Y ($\mu m$)')
            # ax[1].set(xlabel=r'X ($\mu m$)',ylabel=r'Z ($\mu m$)')
            # ax[2].set(xlabel=r'Y ($\mu m$)',ylabel=r'Z ($\mu m$)')
            # pdb.set_trace()

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
        if 'filament/cluster_extent_xyz' in datapath:
            del datapath['filament/cluster_extent_xyz']
        datapath.create_dataset('filament/cluster_extent_xyz', data=std, dtype='f')
# }}}

# cluster_shape_analysis {{{
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
        if 'filament/asphericity' in datapath:
            del datapath['filament/asphericity']
        if 'filament/acylindricity' in datapath:
            del datapath['filament/acylindricity']
        if 'filament/anisotropy' in datapath:
            del datapath['filament/anisotropy']
        if 'filament/rg2' in datapath:
            del datapath['filament/rg2']
        datapath.create_dataset('filament/asphericity', data=asphericity, dtype='f')
        datapath.create_dataset('filament/acylindricity', data=acylindricity, dtype='f')
        datapath.create_dataset('filament/anisotropy', data=anisotropy, dtype='f')
        datapath.create_dataset('filament/rg2', data=rg2, dtype='f')
# }}}

# cluster_nematic_order {{{
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
        if 'filament/cluster_Smax' in datapath:
            del datapath['filament/cluster_Smax']
        if 'filament/cluster_Sxyz' in datapath:
            del datapath['filament/cluster_Sxyz']
        datapath.create_dataset('filament/cluster_Smax', data=Smax, dtype='f')
        datapath.create_dataset('filament/cluster_Sxyz', data=Sxyz, dtype='f')
# }}}

# cluster_msd_diffusion {{{
def cluster_msd_diffusion(FData, labels, dt=0.05, save=False, savepath=None, datapath=None):
    
    box_size = np.copy(FData.config_['box_size'])
    pos = FData.pos_plus_[:,:,-1*labels.shape[1]:]

    cluster_labels = np.unique( labels[ labels > -1].flatten() ).astype('int')
    nc = len(cluster_labels)

    # initialize com position
    pos_cluster = np.zeros((3, labels.shape[1], nc))
    pos_cluster[:] = np.NaN
    # for each cluster compute center of mass
    for jFrame in np.arange(labels.shape[1]):

        for jc in cluster_labels:
            if jc == -1:
                continue

            # calc com
            idx = np.where(labels[:,jFrame]==jc)[0]
            pos_cluster[:,jFrame, jc] = calc_com_pbc( pos[:,idx,jFrame].transpose(), box_size)

    # unfold cluster trajectories
    pos_cluster_unfolded = np.zeros_like( pos_cluster)
    for jc in range(nc):
        pos_cluster_unfolded[:,:,jc] = unfold_trajectory_njit(pos_cluster[:,:,jc], box_size)
            
    # Get the filaments that are condensed throughout
    idxCondensed = np.where(np.all(labels>-1, axis=1))[0]
    pos_unfolded = FData.unfold_trajectories('com')[:,idxCondensed,-1*labels.shape[1]:]

    # Subtract cluster c.o.m
    for jFrame in range(labels.shape[1]):
        for idxc, jc in enumerate(cluster_labels):
            idx_jc = np.where(labels[idxCondensed,jFrame]==jc)[0]
            if len(idx_jc) > 0:
                pos_unfolded[:,idx_jc,jFrame] = pos_unfolded[:,idx_jc,jFrame]-pos_cluster_unfolded[:,jFrame,idxc].reshape(-1,1)

    # plot sample trajectories
    pos_centered = pos_unfolded-pos_unfolded[:,:,0].reshape(pos_unfolded.shape[0],-1,1)
    
    jplot = np.random.choice( pos_centered.shape[1], 5, replace=False)
    fig,ax = plt.subplots()
    ax = plot_multicolored_lines( 
            pos_centered[0,jplot,::5], 
            pos_centered[1,jplot,::5],
            dt*np.arange( pos_centered.shape[-1])[::5],
            ax,
            label='Time (s)',
            alpha=0.3,
            lw=3
            )
    ax.set(xlabel=r'$X (\mu m)$', ylabel=r'$Y (\mu m)$')
    plt.tight_layout()
    # plt.show()
    plt.savefig(savepath.parent / 'filament_motion_within cluster.pdf')
    plt.close()


    # MSD
    timeArray = dt * np.arange(labels.shape[1])
    MSD = calc_msd_fft( pos_unfolded)
    # slopes_mu = np.diff( np.mean(MSD,axis=0) ) / np.diff( timeArray)
    # slopes_sem = np.diff( np.std(MSD,axis=0)/np.sqrt(MSD.shape[0]) ) / np.diff( timeArray)
    # slope_middle = np.array([ slopes_mu[int( len(slopes_mu)/2 ) ], slopes_sem[int( len(slopes_sem)/2 ) ] ])
    # print('MSD slope (center) = {0:.5f} +- {1:.5f} {2}'.format(slope_middle[0], slope_middle[1], r'$\mu m^2 s^{-1}$' ))

    # Plotting
    fig,ax = plt.subplots()
    ax.plot(timeArray, np.nanmean(MSD, axis=0), color='blue', lw=2, label='N = {0}'.format(MSD.shape[0]) )
    ax.fill_between(timeArray,
                    # np.mean(MSD, axis=0) - np.std(MSD, axis=0)/np.sqrt(MSD.shape[0]),
                    # np.mean(MSD, axis=0) + np.std(MSD, axis=0)/np.sqrt(MSD.shape[0]),
                    np.nanmean(MSD, axis=0) - np.nanstd(MSD, axis=0),
                    np.nanmean(MSD, axis=0) + np.nanstd(MSD, axis=0),
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
        if 'filament/cluster/msd' in datapath:
            del datapath['filament/cluster/msd']
        if 'filament/cluster/msd_std' in datapath:
            del datapath['filament/cluster/msd_std']
        if 'filament/cluster/msd_lagtime' in datapath:
            del datapath['filament/cluster/msd_lagtime']
        datapath.create_dataset('filament/cluster/msd', data=np.nanmean(MSD, axis=0), dtype='f')
        datapath.create_dataset('filament/cluster/msd_std', data=np.nanstd(MSD, axis=0), dtype='f')
        datapath.create_dataset('filament/cluster/msd_lagtime', data=timeArray, dtype='f')
# }}}

# condensed_msd_diffusion {{{
def condensed_msd_diffusion(pos, labels, N=400, dt=1, save=False, savepath=None, datapath=None):
    
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
                    # np.mean(MSD, axis=0) - np.std(MSD, axis=0)/np.sqrt(MSD.shape[0]),
                    # np.mean(MSD, axis=0) + np.std(MSD, axis=0)/np.sqrt(MSD.shape[0]),
                    np.mean(MSD, axis=0) - np.std(MSD, axis=0),
                    np.mean(MSD, axis=0) + np.std(MSD, axis=0),
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
        if 'filament/condensed_msd_slope' in datapath:
            del datapath['filament/condensed_msd_slope']
        datapath.create_dataset('filament/condensed_msd_slope', data=slope_middle, dtype='f')
# }}}

# packing_fraction {{{
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
        plt.close()

    if datapath is not None:
        # save ratio to h5py
        if 'filament/condensed_pf' in datapath:
            del datapath['filament/condensed_pf']
        if 'filament/vapor_pf' in datapath:
            del datapath['filament/vapor_pf']
        datapath.create_dataset('filament/condensed_pf', data=pf_c, dtype='f')
        datapath.create_dataset('filament/vapor_pf', data=pf_v, dtype='f')
# }}}

# PlotHistClusterNumbers {{{
def PlotHistClusterNumbers(labels, savepath=None, save=False, datapath=None):
    """ histogram of cluster numbers """

    # for each time, find cluster number and size
    cluster_size = []
    n_clusters = []
    for jframe in np.arange(labels.shape[1]):
        n_clusters_ = len(set(labels[:,jframe])) - (1 if -1 in labels[:,jframe] else 0)
        n_clusters.append(n_clusters_)
        for jc in range(n_clusters_):
            cluster_size.append( np.sum(labels[:,jframe]==jc) )
    
    fig,axs = plt.subplots(1,2,figsize=(8,3))
    axs[0].plot(n_clusters, color='k', lw=2)
    axs[0].set(xlabel='Frame')
    axs[0].set(ylabel='Number of clusters')
    axs[0].set_ylim(bottom=0)
    axs[1].hist(cluster_size, 16, density=True, label='N = {0}, T = {1}'.format(len(cluster_size), labels.shape[1]))
    axs[1].legend()
    axs[1].set(xlabel='Filament number',ylabel='Probability density')
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

    if save and datapath is not None:
        # save to h5py
        if 'filament/num_in_cluster' in datapath:
            del datapath['filament/num_in_cluster']
        if 'filament/num_cluster' in datapath:
            del datapath['filament/num_cluster']
        datapath.create_dataset('filament/num_in_cluster', data=cluster_size, dtype='f')
        datapath.create_dataset('filament/num_cluster', data=n_clusters, dtype='f')
# }}}

# PlotClusterAstralOrder {{{
def PlotClusterAstralOrder(pos_plus, pos_minus, labels, box_size, savepath=None, save=False, datapath=None):
    """ find astral order for clusters 
    
    Astral Order is defined at the ensemble average of the dot product of the filament orientations with the direction of the filament com from the condensate c.o.m
    
    """
    
    # for each time, for each cluster, find order
    astral_order_all = []
    astral_order_time = []
    
    # Loop over time
    for jframe in np.arange(labels.shape[1]):
        n_clusters_ = len(set(labels[:,jframe])) - (1 if -1 in labels[:,jframe] else 0)

        ast = None
        # Loop over clusters
        for jc in range(n_clusters_):

            # index of filaments inside cluster
            idx = labels[:,jframe]==jc
            
            # Find c.o.m
            com = calc_com_pbc(pos_plus[:,idx,jframe].T, box_size)

            # find positions for fialments relative to com
            pplus = pos_plus[:,idx,jframe]-com.reshape(-1,1)
            pminus = pos_minus[:,idx,jframe]-com.reshape(-1,1)
            
            # Apply PBC
            for jdim,bs in enumerate(box_size):
                pplus[jdim,:][ pplus[jdim,:] <= -0.5*bs] += bs
                pplus[jdim,:][ pplus[jdim,:] > 0.5*bs] -= bs
                pminus[jdim,:][ pminus[jdim,:] <= -0.5*bs] += bs
                pminus[jdim,:][ pminus[jdim,:] > 0.5*bs] -= bs

            # find direction of filament center of mass
            f_com = (pplus + pminus)/2
            f_com = f_com / np.linalg.norm(f_com, axis=0)

            # Find orientations
            xi = (pplus - pminus)
            xi = xi/np.linalg.norm( xi, axis=0)

            # Find dot product and take avergae over all filaments
            astral_order_j = np.sum( xi*f_com, axis=0)
            astral_order_all.append( np.mean( astral_order_j) )
            try:
                ast.append( np.mean( astral_order_j) )
            except:
                ast = [ np.mean( astral_order_j) ]

        if ast is not None:
            astral_order_time.append( np.mean(ast) )
        else:
            astral_order_time.append( np.NaN)
            
    fig,axs = plt.subplots(1,2,figsize=(8,3))
    axs[0].plot(astral_order_time, color='k', lw=2)
    axs[0].set(xlabel='Frame')
    axs[0].set(ylabel='Astral order')
    axs[0].set_ylim(bottom=0, top=1.0)

    axs[1].hist(astral_order_all, 24, density=True )
    axs[1].set(xlabel='Astral order',ylabel='Probability density')
    axs[1].set_xlim(left=0, right=1.0)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

    if save and datapath is not None:
        # save to h5py
        if 'filament/astral_order_time' in datapath:
            del datapath['filament/astral_order_time']
        datapath.create_dataset('filament/astral_order_time', data=astral_order_time, dtype='f')
        # datapath.create_dataset('filament/', data=n_clusters, dtype='f')
# }}}

# PlotCondensedLocalOrder {{{
def PlotCondensedLocalOrder(FData, labels, N=300, savedir=None, datapath=None):
    """ find Local Polar Order of condensed filaments """
    
    # Check if Local Order has been found. If not, raise error
    # if not FData.local_order_calculated:
        # raise Exception('Please first evaluate FData.LocalStructure()')

    # Define histogram bins
    bins_lpo = np.linspace(-1,1,50)
    bins_lpo_cen = 0.5*(bins_lpo[1:]+bins_lpo[:-1])
    bins_lno = np.linspace(0,1,25)
    bins_lno_cen = 0.5*(bins_lno[1:]+bins_lno[:-1])

    # Extract labels for condensed filaments
    labels_specific = labels[:,-1*N:]
    # Local Polar Order
    lpo = FData.local_polar_order[:,-1*N:].flatten()[labels_specific.flatten() == 0]
    # Local Nematic Order
    lno = FData.local_nematic_order[:,-1*N:].flatten()[labels_specific.flatten() == 0]
    
    lpo_pdf = np.histogram(lpo.flatten(), density=True, bins=bins_lpo)[0]
    lno_pdf = np.histogram(lno.flatten(), density=True, bins=bins_lno)[0]


    # Plot LPO, LNO
    if savedir:
        fig,(ax0, ax1) = plt.subplots(1,2, figsize=(8,3))
        sns.histplot(lno.flatten(), kde=True,
             bins=bins_lno, color = 'darkblue',
             edgecolor='black',
             line_kws={'linewidth': 4}, ax=ax0)
        ax0.set_xlabel('Local nematic order')
        sns.histplot(lpo.flatten(), kde=True,
             bins=bins_lpo, color = 'darkblue',
             edgecolor='black',
             line_kws={'linewidth': 4}, ax=ax1)
        ax1.set_xlabel('Local polar order')
        plt.tight_layout()
        plt.savefig(savedir/'condensed_local_order.pdf', bbox_inches="tight")
        plt.close()

    # save to h5py
    if datapath is not None:
        if 'filament/condensed_lpo' in datapath:
            del datapath['filament/condensed_lpo']
        if 'filament/condensed_lpo_bin_centers' in datapath:
            del datapath['filament/condensed_lpo_bin_centers']
        if 'filament/condensed_lpo_bin_edges' in datapath:
            del datapath['filament/condensed_lpo_bin_edges']
        if 'filament/condensed_lno' in datapath:
            del datapath['filament/condensed_lno']
        if 'filament/condensed_lno_bin_centers' in datapath:
            del datapath['filament/condensed_lno_bin_centers']
        if 'filament/condensed_lno_bin_edges' in datapath:
            del datapath['filament/condensed_lno_bin_edges']
        datapath.create_dataset('filament/condensed_lpo', data=lpo_pdf, dtype='f')
        datapath.create_dataset('filament/condensed_lpo_bin_centers', data=bins_lpo_cen, dtype='f')
        datapath.create_dataset('filament/condensed_lpo_bin_edges', data=bins_lpo, dtype='f')
        datapath.create_dataset('filament/condensed_lno', data=lno_pdf, dtype='f')
        datapath.create_dataset('filament/condensed_lno_bin_centers', data=bins_lno_cen, dtype='f')
        datapath.create_dataset('filament/condensed_lno_bin_edges', data=bins_lno, dtype='f')

# }}}

# PlotCondensedTrajectories {{{
def PlotCondensedTrajectories(FData, labels, N=20, savepath=None):

    # Get the filaments that are condensed for the last 200 frames
    idx = np.sum( labels[:,-1*N:], axis=1) == 0
    pos = FData.pos_plus_[:,idx,-1*N:]
    pos = FData.unfold_trajectories('plus')[:,idx,-1*N:]

    fig,ax = plt.subplots()
    for jidx in np.random.choice(pos.shape[1], N):
        ax.plot(pos[0,jidx,:]-pos[0,jidx,0], 
                pos[1,jidx,:]-pos[1,jidx,0], 
                alpha=0.4)

    ax.set(xlabel='X', ylabel='Y')
    # ax.set_xlim(left=0, right=FData.config_['box_size'][0])
    # ax.set_ylim(bottom=0, top=FData.config_['box_size'][1])
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    plt.close()
# }}}

# TrackClusterLabels {{{
def TrackClusterLabels(labels):
    from scipy import stats

    nFrames = labels.shape[1]
    nParticles = labels.shape[0]

    labels_tracked = np.zeros_like(labels)
    labels_tracked[:] = -1
    labels_tracked[:,0] = labels[:,0]

    max_lab = int(np.max(labels.flatten())-1)
    for jFrame in np.arange(1,nFrames):

        # Get cluster labels in this frame
        cluster_labs = np.unique(labels[:,jFrame])

        # Loop over clusters and reassign labels accordingly
        for jc in cluster_labs:
            if jc == -1:
                continue

            idx = np.where(labels[:,jFrame]==jc)[0]
            
            #Find cluster index in past frame
            labs = labels_tracked[idx,:jFrame][:,-3:]
            labs = labs[labs != -1]
            if len(labs) == 0:
                old_label = max_lab+1
                max_lab = old_label
            else:
                old_label = stats.mode( labs)[0][0]
            
            # Set new labels
            labels_tracked[idx,jFrame] = old_label

            # if jFrame == 195 and jc==8:
                # pdb.set_trace()

    return labels_tracked
# }}}

# PlotClusterMotion {{{
def PlotClusterMotion(pos, labels, box_size, savepath=None):

    # find unique clusters
    cluster_labels = np.unique( labels[ labels > -1].flatten() ).astype('int')
    nc = len(cluster_labels)

    # initialize com position
    pos_cluster = np.zeros((3, labels.shape[1], nc))
    pos_cluster[:] = np.NaN

    # for each cluster compute center of mass
    for jFrame in np.arange(labels.shape[1]):

        for jc in cluster_labels:
            if jc == -1:
                continue

            # calc com
            idx = np.where(labels[:,jFrame]==jc)[0]
            pos_cluster[:,jFrame, jc] = calc_com_pbc( pos[:,idx,jFrame].transpose(), box_size)
            

    if savepath:
        fig,ax = plt.subplots(1,3, figsize=(12,4))
        colors = sns.color_palette("husl", nc)
        for jc in np.arange(nc):
            
            pp = unfold_trajectory_njit(pos_cluster[:,:,jc], box_size)
            # plot 
            # ax[0].plot( pos_cluster[0,:,jc], pos_cluster[1,:,jc], color=colors[jc], 
                    # label=jc, alpha=0.5, linestyle = 'None', marker='D')
            # ax[1].plot( pos_cluster[0,:,jc], pos_cluster[2,:,jc], color=colors[jc], 
                    # label=jc, alpha=0.5, linestyle = 'None', marker='D')
            # ax[2].plot( pos_cluster[1,:,jc], pos_cluster[2,:,jc], color=colors[jc], 
                    # label=jc, alpha=0.5, linestyle = 'None', marker='D')
            ax[0].plot( pp[0,:], pp[1,:], color=colors[jc], 
                    label=jc, alpha=0.5) 
            ax[1].plot( pp[0,:], pp[2,:], color=colors[jc], 
                    label=jc, alpha=0.5)
            ax[2].plot( pp[1,:], pp[2,:], color=colors[jc], 
                    label=jc, alpha=0.5)
            # ax[0].plot( masked_datax, masked_datay, color=colors[jc], label=jc, alpha=1.0)
            # ax[1].plot( masked_datax, masked_dataz, color=colors[jc], label=jc, alpha=1.0)
            # ax[2].plot( masked_datay, masked_dataz, color=colors[jc], label=jc, alpha=1.0)
        ax[0].legend()
        ax[0].set(xlabel=r'X ($\mu m$)',ylabel=r'Y ($\mu m$)')
        ax[1].set(xlabel=r'X ($\mu m$)',ylabel=r'Z ($\mu m$)')
        ax[2].set(xlabel=r'Y ($\mu m$)',ylabel=r'Z ($\mu m$)')
        plt.tight_layout()
        plt.savefig(savepath)
        plt.close()

# }}}

# PlotResidenceTimes {{{
def PlotResidenceTimes(labels, N=400, dt=0.05, savepath=None, datapath=None):
    """ Reisidence time is the time spent by a particle in the condensed state """
    
    # Only analyze the last N frames
    labels_choose = labels[:,-1*N:] 
   
    # Residence times
    residence_times = []

    bins = np.linspace(0,20,100)
    bins_c = 0.5*(bins[1:]+bins[:-1])

    # Loop over filaments, and find residence times
    for jfil in np.arange(labels_choose.shape[1]):

        # Find derivative of states (changes are now at 1 and -1)
        # Start = 1, and exit = -1
        der = np.diff(labels_choose[jfil, :])

        # Find non-zero entries and their indices. If no entries, continue
        # If one entry, then continue too, since we have a start and not an end
        vals = der[der != 0]
        if len(vals) <= 1:
            continue

        # Find difference between indices to find time spent in states
        idx = np.where( der != 0)[0]
        time_spent = np.diff(idx)
        state_occupied = labels_choose[ jfil, idx[1:]]
        
        # Condensed states occupied
        res = time_spent[state_occupied == 0]
        for jres in res:
            residence_times.append(dt*jres)
    
    if savepath is not None:
        fig,ax = plt.subplots()
        # ax.hist(residence_times, 24, density=True, label='N = {0}'.format(len(residence_times) ) )
        sns.histplot(residence_times, kde=True,
             bins=bins, color = 'darkblue',
             edgecolor='black',
             line_kws={'linewidth': 4}, ax=ax)
        ax.set(xlabel='Residence time (s)',ylabel='Probability density')
        ax.set_xlim(left=0)
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(savepath)
        plt.close()
        
    if datapath is not None:
        # save to h5py
        if 'filament/condensed_residence_time' in datapath:
            del datapath['filament/condensed_residence_time']
        datapath.create_dataset('filament/condensed_residence_time', data=residence_times, dtype='f')
    # return residence_times
# }}}

# CalcBundleTwist {{{
def CalcBundleTwist( FData, labels, N=25, savepath=None, datapath=None):

    frames = np.arange(FData.nframe_-N, FData.nframe_)
    
    # get constantly condensed filaments for last N frames
    idx = np.sum( labels[:,frames], axis=1) == 0
    n_cond = np.sum(idx)
    # idx = np.arange(0,FData.nfil_)
    # n_cond = len(idx)
    
    # for each frame, calculate bundle twist
    twist = np.zeros( (n_cond, N) )
    rdist = np.zeros( (n_cond, N) )
    for jframe,cframe in enumerate(frames):
        twist[:,jframe], rdist[:,jframe] = calc_bundle_twist(
                FData.get_com()[:,idx,cframe], 
                FData.orientation_[:,idx,cframe], 
                np.copy(FData.config_['box_size']) )
        
    if datapath is not None:
        # save to h5py
        if 'filament/condensed_twist' in datapath:
            del datapath['filament/condensed_twist']
        datapath.create_dataset('filament/condensed_twist', data=twist.flatten(), dtype='f')
        # datapath.create_dataset('filament/', data=n_clusters, dtype='f')

    return twist,rdist
# }}}
        
# PlotHistBundleTwist_vs_RadialDist {{{
def PlotHistBundleTwist_vs_RadialDist( twist,rdist, savepath=None):

    # evaluate on a regular grid
    rgrid = np.linspace(0,0.2,40)
    tgrid = np.linspace(-0.3,0.3,40)

    # Plot the result as an image
    if savepath is not None:
        fig,ax = plt.subplots()
        im = ax.hist2d(rdist.flatten(), twist.flatten(),
               bins = [rgrid, tgrid], density=True,
               cmap ="viridis")
        fig.colorbar(im[3], ax=ax, label="Probability density")
        ax.set(xlabel=r'Radial distance $(\mu m)$', ylabel='Twist parameter')
        plt.tight_layout()
        plt.savefig(savepath, bbox_inches="tight")
        plt.close()
# }}}
         
# Plot2DBundleTwist_NematicBasis{{{
def Plot2DBundleTwist_NematicBasis(FData, labels, savepath=None):

    # Get condensed filament indices in last frame
    idx = labels[:,-1] == 0

    # Get filament com positions 
    pos = FData.get_com()[:,idx,-1]
    
    # Find twist
    twist,_ = calc_bundle_twist(pos, 
            FData.orientation_[:,idx,-1], 
            np.copy(FData.config_['box_size']))

    # Find condensate center of mass
    com = calc_com_pbc( pos.T, FData.config_['box_size'])
    
    # Find filament positions relative to center of mass
    pos_rel = pos - com.reshape(-1,1)

    # Find nematic basis vectors
    nbasis = calc_nematic_basis(FData.orientation_[:,idx,-1])

    # Find new filament positions in this nematic basis
    pos_rot = np.linalg.solve( nbasis, pos_rel)

    if savepath is not None:

        # Plot filaments in X' Y' frame
        fig,ax = plt.subplots(figsize=(4,3))
        col0 = twist
        norm0 = mpl.colors.Normalize(vmin=-0.3, vmax=0.3)
        cmap0 = mpl.cm.ScalarMappable(norm=norm0, cmap=mpl.cm.jet)
        cmap0.set_array([])
        ax.scatter(
                pos_rot[0,:],
                pos_rot[1,:],
                c=cmap0.to_rgba(col0), alpha=0.5)
        fig.colorbar(cmap0, label=r'Twist $\chi$', ax=ax)
        ax.set(xlabel=r'$X^\prime (\mu m)$', ylabel=r'$Y^\prime (\mu m)$')
        plt.tight_layout()
        plt.savefig(savepath, bbox_inches="tight")
        plt.close()
# }}}

# Plot3DBundleTwist{{{
def Plot3DBundleTwist(FData, labels, nematic_basis=False,savepath=None):

    # Get condensed filament indices in last frame
    idx = labels[:,-1] == 0

    # Get filament positions 
    p0 = FData.pos_minus_[:,idx,-1]
    p1 = FData.pos_plus_[:,idx,-1]
    pos = FData.get_com()[:,idx,-1]

    # Find condensate center of mass
    com = calc_com_pbc( pos.T, FData.config_['box_size'])

    # Find filament positions relative to center of mass
    pos_rel = pos - com.reshape(-1,1)
    p0_rel = p0 - com.reshape(-1,1)
    p1_rel = p1 - com.reshape(-1,1)
    
    # Find twist
    twist,_ = calc_bundle_twist(pos, 
            FData.orientation_[:,idx,-1], 
            np.copy(FData.config_['box_size']))

    if nematic_basis:
        # Find nematic basis vectors
        nbasis = calc_nematic_basis(FData.orientation_[:,idx,-1])

        # Find new filament positions in this nematic basis
        p0_rel = np.linalg.solve( nbasis, p0_rel)
        p1_rel = np.linalg.solve( nbasis, p1_rel)

    Plot3DBundle_ColorByVec(p0_rel, p1_rel, FData.config_['box_size'],
        color_vec=twist, color_range=[-0.3,0.3],
        color_label=r'Twist $\chi$',
        savepath=savepath)
# }}}

# Plot3DBundle_ColorByVec {{{
def Plot3DBundle_ColorByVec(p0, p1, box_size,
        basis=None,color_vec=None, color_range=None,color_label='Color',
        savepath=None):

    # Get filament com positions 
    pos = (p0 + p1 )/2
    
    # Find condensate center of mass
    com = calc_com_pbc( pos.T, box_size)
    
    if color_vec is not None:
        if color_range is None:
            color_range=[np.min(color_vec), np.max(color_vec)]
        col0 = color_vec 
        norm0 = mpl.colors.Normalize(vmin=color_range[0], vmax=color_range[1])
        cmap0 = mpl.cm.ScalarMappable(norm=norm0, cmap=mpl.cm.jet)
        cmap0.set_array([])
        col=cmap0.to_rgba(col0).T
    else:
        col = np.zeros((4,p0.shape[-1]))
        col[0,:] = 1
        col[-1,:] = 1

    if savepath is not None:

        # Plot filaments in X' Y' frame
        fig,(ax0,ax1,ax2) = plt.subplots(1,3,figsize=(12,3))
        
        for jfil in np.arange(p0.shape[-1]):
            ax0.plot(
                    [p0[0,jfil],p1[0,jfil]],
                    [p0[1,jfil],p1[1,jfil]],
                    c=col[:,jfil], alpha=0.5)
            ax1.plot(
                    [p0[0,jfil],p1[0,jfil]],
                    [p0[2,jfil],p1[2,jfil]],
                    c=col[:,jfil], alpha=0.5)
            ax2.plot(
                    [p0[1,jfil],p1[1,jfil]],
                    [p0[2,jfil],p1[2,jfil]],
                    c=col[:,jfil], alpha=0.5)
        if color_vec is not None:
            fig.colorbar(cmap0, label=color_label, ax=ax2)
        ax0.set(xlabel=r'$X (\mu m)$', ylabel=r'$Y (\mu m)$')
        ax1.set(xlabel=r'$X (\mu m)$', ylabel=r'$Z (\mu m)$')
        ax2.set(xlabel=r'$Y (\mu m)$', ylabel=r'$Z (\mu m)$')
        plt.tight_layout()
        plt.savefig(savepath, bbox_inches="tight")
        plt.close()
# }}}

# condensed_aspect_ratio {{{
def condensed_aspect_ratio(pos, orientation, labels, box_size, savepath=None, save=False, datapath=None):
    """
    To do this:
    1) Find condensed c.o.m, and relative filament positions
    2) Find gyration tensor
    3) Find basis vectors
    4) Transform filament positions to new basis
    5) Find extents in each dimension
    """

    sigma = np.zeros((3,pos.shape[2]))

    # frames
    frames = np.arange(pos.shape[2])

    for jf in frames:
        cpos = pos[:,labels[:,jf]==0, jf]

        # find center of mass
        com = calc_com_pbc(cpos.T, box_size)

        # find periodic distance to com for each point
        xyz = np.zeros_like(cpos)
        for jd in range(cpos.shape[0]):
            cc = cpos[jd,:]-com[jd]
            Lc = box_size[jd]
            cc[ cc >= Lc/2] -= Lc
            cc[ cc < -Lc/2] += Lc
            xyz[jd,:] = cc

        # sum over outer products
        outs = np.zeros((3,3))
        for jc in range(cpos.shape[1]):
            outs+=np.outer(xyz[:,jc], xyz[:,jc])
        outs = outs/cpos.shape[1]

        # diagonalize
        vals, vecs = np.linalg.eig(outs)

        # sort vals (ascending order)
        inds = vals.argsort()
        vals = vals[inds]
        vecs = vecs[:,inds]
        
        basis = vecs[:,::-1]

        # Find new filament positions in this basis
        pos_rot = np.linalg.solve( basis, cpos)
        
        for jdim in range(3):
            sigma[jdim,jf] = np.max(pos_rot[jdim,:]) - np.min(pos_rot[jdim,:])

    if save:
        colors = sns.color_palette("husl", 3)
        fig,(ax0,ax1) = plt.subplots(1,2,figsize=(8,3))
        labs = [r'$\sigma_0$',r'$\sigma_1$',r'$\sigma_2$']
        for jd in range(3):
            ax0.plot( sigma[jd,:], label=labs[jd], linewidth=2,color=colors[jd])
        ax0.legend(frameon=False)
        ax0.set(xlabel=r'$\sigma (\mu m$)',ylabel='Frame')
        
    ratio = sigma[0,:]/(0.5*(sigma[1,:]+sigma[2,:]))
    ax1.plot(ratio, color='k', lw=2)
    ax1.set(xlabel='Frame')
    ax1.set(ylabel=r'Aspect ratio $2\sigma_0 / (\sigma_1 + \sigma_2)$')
    ax1.set_ylim(bottom=0.0, top=5.02)
    ax1.axhline(ratio[-1],color='k', linestyle='dotted')
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()
    
    # if datapath is not None:
        # # save ratio to h5py
        # if 'filament/cluster_extent_xyz' in datapath:
            # del datapath['filament/cluster_extent_xyz']
        # datapath.create_dataset('filament/cluster_extent_xyz', data=sigma, dtype='f')
# }}}
