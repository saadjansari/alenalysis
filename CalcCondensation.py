from numba import njit
import numpy as np
from matplotlib import pyplot as plt
import decorators
from CalcMobility import calc_mobility
from CalcNumXlinks import calc_num_xlink_filament
import pdb
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def PlotFilamentCondensation(FData, XData, savepath):
    """ Plot the ratio of condensed filaments"""
    
    print('Ratio of condensed filaments...') 

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
    df = pd.DataFrame(data, columns=names)

    # Cluster: get kmeans label 
    labels = get_label_from_kmeans(df, nx.shape)

    # Plot time averaged label
    PlotTimeAvgLabelHist(labels, savepath.parent / 'clustering_time_avg_label.pdf')

    # Plot ratio of condensed filaments over time
    PlotRatioCondensedFilaments(labels, savepath.parent / 'clustering_ratio_condensed.pdf')


def get_label_from_kmeans(df, data_shape, n_clusters=2):
    """ get label from kmeans clustering """

    # Clustering Algorithm KMeans
    data = df.to_numpy()
    scaler = StandardScaler()
    X_std = scaler.fit_transform(data)

    print('KMeans clustering: N_clusters = {}'.format(n_clusters))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_std)
    labels = kmeans.labels_
    
    if kmeans.cluster_centers_[0,0] > kmeans.cluster_centers_[1,0]:
        # Switch labels
        labels_cpy = labels.copy()
        labels_cpy[ labels == 0] = 1
        labels_cpy[ labels == 1] = 0
        labels = labels_cpy

    labels = np.reshape(labels, data_shape)
    return labels

def PlotTimeAvgLabelHist(labels, savepath):
    fig,ax = plt.subplots()
    ax.hist(np.mean(labels, axis=1), bins=50)
    ax.set(yscale='log', title='Time Averaged Label', xlabel='Mean Label', ylabel='Count')
    plt.tight_layout()
    plt.savefig(savepath)

def PlotRatioCondensedFilaments(labels, savepath):

    ratio = np.zeros( labels.shape[1])
    for jt in range(labels.shape[1]):
        ratio[jt] = np.sum(labels[:,jt] == 1) / labels.shape[0]

    fig,ax = plt.subplots()
    ax.plot(ratio, color='blue', lw=2)
    ax.set(xlabel='Frame')
    ax.set(ylabel='Ratio of Condensed Fil')
    ax.set_ylim(bottom=-0.02, top=1.02)
    plt.tight_layout()
    plt.savefig(savepath)

    print('Condensed Fraction = {0:.2f} +- {1:.3f}'.format(
        np.mean(ratio[-50:]), np.std(ratio[-50:])))

# def PlotLabelCounts(idx, savepath):

    # # # 
    # # # Form clusters by incorporating time
    # # thresh = [0.02,0.98]
    # # idx_0 = np.where(np.mean(labels, axis=1) <= thresh[0])[0]
    # # idx_1 = np.where(np.mean(labels, axis=1) >= thresh[1])[0]
    # # idx_2 = np.where( (np.mean(labels, axis=1) < thresh[1])*(np.mean(labels, axis=1) > thresh[0]) )[0]

    # # # temp count check
    # # print('Group 1: {0}\nGroup 2: {1}\nGroup 3: {2}\nTotal: {3}\nTotal Expected: {4}'.format(
        # # len(idx_0), len(idx_1),len(idx_2), len(idx_0)+len(idx_1)+len(idx_2), pos.shape[1]))
    # # idx = np.zeros(pos.shape[1], dtype=int)
    # # idx[idx_0] = 0:
    # # idx[idx_1] = 1
    # # idx[idx_2] = 2
    # fig,ax=plt.subplots()
    # occurance = [len(idx_0), len(idx_1), len(idx_2)]
    # ax.bar( np.arange(len(occurance)), occurance, align='center', alpha=1, color='y')
    # ax.set(ylabel='Count', title='Label Counts\nMean Label Threshold = {}'.format(thresh))
    # ax.set_xticks(np.arange(len(occurance)))
    # plt.tight_layout()
    # plt.savefig( simpath / 'npy/label_counts.pdf')
    
    # return kmeans.labels_
