import numpy as np
from random import sample
import matplotlib.pyplot as plt
import pdb
from src.unfold_trajectories import *
from src.CalcOrderParametersLocal import * 
from src.CalcPackingFraction import * 
from src.CalcOverlaps import minDistBetweenAllFilaments

# DataSeries {{{
class DataSeries:
    """Class for handling data"""
    def __init__(self, gid, pos_minus, pos_plus, config):
        self.gid_ = gid                     # size N x T
        self.pos_minus_ = pos_minus         # size 3 x N x T
        self.pos_plus_ = pos_plus           # size 3 x N x T
        self.config_ = config               # dictionary of configuration parameters
        self.nframe_ = self.pos_minus_.shape[2]
        self.ndim_ = self.pos_minus_.shape[0]

    # Center of Mass
    def get_com(self):
    # get center of mass of filaments

        com = 0.5*(self.pos_plus_ + self.pos_minus_)
        # # Apply PBC
        # pdb.set_trace()
        # for jdim,bs in enumerate(self.config_['box_size']):
            # com[jdim,:,:][com[jdim,:,:] <= -0.5*bs] += bs
            # com[jdim,:,:][com[jdim,:,:] > 0.5*bs] -= bs
        return com
    
    # Unfold trajectories {{{
    def unfold_trajectory(self, obj_index, which_end):
        # unfolded crds via the nearest image convention

        # Extract the relevant object and its end
        # Note that we transpose so that dimensional order is nTime x nDim
        if which_end == 'minus':
            crds = self.pos_minus_[:,obj_index, :]
        elif which_end == 'plus':
            crds = self.pos_plus_[:,obj_index, :]
        elif which_end == 'com':
            crds = self.get_com()[:,obj_index, :]
        else:
            raise Exception("which_end must be 'plus', 'minus', or 'com'")

        crds_unfolded = unfold_trajectory_njit(crds,self.config_['box_size'])
        return crds_unfolded

    def unfold_trajectories(self, which_end):
        """
        Inputs:
        -------------
        pos: 3D float array
            Size is nDim x nFil x nTime
        """
        if which_end == 'minus':
            crds = self.pos_minus_
        elif which_end == 'plus':
            crds = self.pos_plus_
        elif which_end == 'com':
            crds = self.get_com()
        else:
            raise Exception("which_end must be 'plus','minus', or 'com'")

        pos_unfolded = unfold_trajectories_njit(crds,self.config_['box_size'])
        return pos_unfolded
    # }}}

# DataSeries }}}

# FilamentSeries {{{
class FilamentSeries(DataSeries):
    """Class for handling data related to filaments."""
    def __init__(self, gid, pos_minus, pos_plus, orientation, config):
        super().__init__(gid, pos_minus, pos_plus, config)
        self.orientation_ = orientation     # size 3 x N x T
        self.nfil_ = self.pos_plus_.shape[1]
        self.min_dist_calculated = False
        self.local_order_calculated = False

    # Methods:
    # Local Structure {{{
    def CalcLocalStructure(self, N=100):
        self.CalcMinDist(N)
        self.CalcLocalOrder(N)
        # self.CalcLocalPackingFraction(N)

    # Local Order {{{
    def CalcLocalOrder(self, N, max_dist_ratio=50):

        self.local_polar_order_, self.local_nematic_order_ = calc_local_order(
                self.orientation_[:,:,-1*N:], 
                self.min_dist_)
        self.local_order_calculated = True
    # }}}
    
    # Local Packing Fraction {{{
    def CalcLocalPackingFraction(self, N, max_dist_ratio=50):
        self.local_packing_fraction_ = calc_local_packing_fraction(
                self.pos_minus_[:,:,-1*N:],
                self.pos_plus_[:,:,-1*N:], 
                self.config_['diameter_fil'], 
                self.config_['box_size'])
    # }}}

    # Min Dist {{{
    def CalcMinDist(self, N=100):
        print('Computing minimum distances for last {0} frames'.format(N))
        self.min_dist_ = np.zeros((self.nfil_, self.nfil_, N))

        for jframe, cframe in enumerate( range(self.nframe_-N,self.nframe_) ):

            print('Frame = {0}/{1}'.format(cframe,self.nframe_), 
                    end='\r', flush=True )
        
            # overlap matrix (normalized by filament diameter)
            self.min_dist_[:,:,jframe] = minDistBetweenAllFilaments( 
                    self.pos_minus_[:,:,cframe], 
                    self.pos_plus_[:,:,cframe], 
                    self.pos_minus_[:,:,cframe], 
                    self.pos_plus_[:,:,cframe]) / self.config_['diameter_fil']
        print('Frame = {0}/{0}'.format(self.nframe_))
        self.min_dist_calculated = True
    # }}}
    # }}} 
    
    # Plot trajectories {{{
    def plot_trajectories(self, savepath, fil_index=None, n_samples=10, which_end='com', **kwargs):

        # If indices to plot not provided, create a random sample of indices
        if fil_index is None:
            fil_index = sample( list(np.arange(self.nfil_)), n_samples)

        # Time array
        times = self.config_['time_snap']*np.arange(self.nframe_)

        # Dimension labels
        dim_names = [r'$X / \mu m$', r'$Y / \mu m$', r'$Z / \mu m$']

        # Plotting
        fig, axs = plt.subplots(3,1, figsize=(12,12), sharex=True)
        for jfil in fil_index:

            # Unfold trajectory of relevant xlinker
            pos_fil = self.unfold_trajectory(jfil, which_end)

            for jdim in range(self.ndim_):
                axs[jdim].plot( times, pos_fil[jdim, :], **kwargs)

        # Labels
        for jdim in range(self.ndim_):
            axs[jdim].set(ylabel=dim_names[jdim])
        axs[-1].set(xlabel='Time / s')
        axs[0].set(title='Sample Filament Trajectories')

        # Save plot
        plt.tight_layout()
        plt.savefig(savepath)
        plt.close()
    # }}}

    # }}}

# CrosslinkerSeries {{{
class CrosslinkerSeries(DataSeries):
    """Class for handling data related to crosslinkers."""
    def __init__(self, gid, pos_minus, pos_plus, link0, link1, config):
        super().__init__(gid, pos_minus, pos_plus, config)
        self.gid_ = gid                     # size N x T
        self.link0_ = link0                 # size N x T
        self.link1_ = link1                 # size N x T
        self.nxlink_ = self.pos_plus_.shape[1]

    # Methods
    # Binding State {{{
    def get_bind_state(self, xlink_index=None):
        """
        Binding State:
            0 --> Free / Unbound
            1 --> Singly-Bound
            2 --> Doubly-Bound
        """

        if xlink_index is None:
            xlink_index = np.arange(self.nxlink_)

        bind_state = np.zeros((len(xlink_index), self.nframe_))
        for jidx, jxlink in enumerate(xlink_index):
            for jt in np.arange(bind_state.shape[1]):
                if self.link0_[jxlink, jt] > -1 and self.link1_[jxlink, jt] > -1:
                    bind_state[jidx,jt] = 2
                elif self.link0_[jxlink, jt] == -1 and self.link1_[jxlink, jt] == -1:
                    bind_state[jidx,jt] = 0
                else:
                    bind_state[jidx,jt] = 1
        return bind_state
    # Binding State }}}

    # Energy and Length {{{
    def get_length(self):
        # get length of doubly-bound crosslinkers
        # List (times)  of lists (crosslinker lengths)

        # All lengths
        lens_all = np.linalg.norm(self.pos_plus_-self.pos_minus_, axis=0)
        lens_doubly = []
        
        # Get bind-states
        bind_state = self.get_bind_state()

        for jt in np.arange(self.nframe_):
            lens_doubly.append( [this_len for bs,this_len in zip(bind_state[:,jt],lens_all[:,jt]) if bs==2] )
        return lens_doubly 

    def get_energy(self):
        # get spring energy of doubly-bound crosslinkers
        # List (times)  of lists (crosslinker lengths)
        kappa = self.config_['kappa'][0]
        rest_len = self.config_['rest_length'][0]

        lens = self.get_length()
        energies = []
        for jt in range(self.nframe_):
           energies.append( [(kappa/2)*(jlen-rest_len)**2 for jlen in lens[jt] ] )
        return energies 

    # Plot length
    def plot_length_mean_vs_time(self, savepath, **kwargs):

        print('Plotting Mean Crosslinker Length Vs Time...')
        # Time array
        times = self.config_['time_snap']*np.arange(self.nframe_)

        # List of lengths
        len_raw = self.get_length()
        len_array = np.zeros( (self.nframe_, 2) )

        # Plotting
        fig, ax = plt.subplots()
        for jt in range(self.nframe_):
            # Mean and std of crosslinker length
            if len( len_raw[jt]) > 0:
                len_array[jt,:] = [np.mean(len_raw[jt]), np.std(len_raw[jt])]
            else:
                len_array[jt,:] = [np.nan, np.nan]

        ax.errorbar(times, len_array[:,0], yerr=len_array[:,1], 
                marker='.', ms=1, mew=0.5, mfc="None", alpha=0.5,
                lw=1, linestyle = 'None', elinewidth=0.5, capsize=1,
                **kwargs)

        # Labels
        ax.set(ylabel=r'Length / $\mu m$', xlabel='Time / s')
        ax.set_ylim(bottom=-0.002)

        # Save plot
        plt.tight_layout()
        plt.savefig(savepath)
        plt.close()

    # Plot energies 
    def plot_energy_mean_vs_time(self, savepath, **kwargs):

        print('Plotting Crosslinker Mean Energy Vs Time...')
        # Time array
        times = self.config_['time_snap']*np.arange(self.nframe_)

        # List of energies
        eng_raw = self.get_energy()
        eng_array = np.zeros( (self.nframe_, 2) )

        # Plotting
        fig, ax = plt.subplots()
        for jt in range(self.nframe_):
            # Mean and std of crosslinker energy
            if len( eng_raw[jt]) > 0:
                eng_array[jt,:] = [np.mean(eng_raw[jt]), np.std(eng_raw[jt])]
            else:
                eng_array[jt,:] = [np.nan, np.nan]


        ax.errorbar(times, eng_array[:,0], yerr=eng_array[:,1], 
                marker='.', ms=1, mew=0.5, mfc="None", alpha=0.5,
                lw=1, linestyle = 'None', elinewidth=0.5, capsize=1,
                **kwargs)

        # Labels
        ax.set(ylabel=r'Energy / pN', xlabel='Time / s')
        ax.set_ylim(bottom=-0.002)

        # Save plot
        plt.tight_layout()
        plt.savefig(savepath)
        plt.close()
    # Lengths }}}

    # Plot trajectories {{{
    def plot_trajectories(self, savepath, which_end='com', xlink_index=None, n_samples=10, **kwargs):

        # If indices to plot not provided, create a random sample of indices
        if xlink_index is None:
            xlink_index = sample( list(np.arange(self.nxlink_)), n_samples)

        # Time array
        times = self.config_['time_snap']*np.arange(self.nframe_)

        # Dimension labels
        dim_names = [r'$X / \mu m$', r'$Y / \mu m$', r'$Z / \mu m$']

        # Plotting
        fig, axs = plt.subplots(3,1, figsize=(12,12), sharex=True)
        for jxlink in xlink_index:

            # Unfold trajectory of relevant xlinker
            pos_xlink = self.unfold_trajectory(jxlink, which_end)

            for jdim in range(self.ndim_):
                axs[jdim].plot( times, pos_xlink[jdim, :], **kwargs)

        # Labels
        for jdim in range(self.ndim_):
            axs[jdim].set(ylabel=dim_names[jdim])
        axs[-1].set(xlabel='Time / s')
        axs[0].set(title='Sample Crosslinker Trajectories')

        # Save plot
        plt.tight_layout()
        plt.savefig(savepath)
        plt.close()
    # Plot trajectories }}}
# }}}

# AddCorrelationsToDataSeries {{{
def AddCorrelationsToDataSeries(DSeries, simpath):
    
    # Contact Number
    with open(simpath / 'contact_number.npy', 'rb') as f:
        contact_number = np.load(f)

    # Local Order
    with open(simpath / 'local_order.npy', 'rb') as f:
        local_polar_order = np.load(f)
        local_nematic_order = np.load(f)

    # Initialize to nans
    cn = np.zeros( (contact_number.shape[0], DSeries.nframe_) )
    lpo = np.zeros( (local_polar_order.shape[0], DSeries.nframe_) )
    lno = np.zeros( (local_nematic_order.shape[0], DSeries.nframe_) )
    cn[:] = np.NaN
    lpo[:] = np.NaN
    lno[:] = np.NaN

    if contact_number.shape[1] == DSeries.nframe_:
        print('Number of processed correlation data matches number of vtk files')
        cn = contact_number
        lpo = local_polar_order
        lno = local_nematic_order
    elif contact_number.shape[1] > DSeries.nframe_:
        print('Number of processed correlation data is greater than number of vtk files')
        cn = contact_number[:,:DSeries.nframe_]
        lpo = local_polar_order[:,:DSeries.nframe_]
        lno = local_nematic_order[:,:DSeries.nframe_]
    elif contact_number.shape[1] < DSeries.nframe_:
        print('Number of processed correlation data is less than number of vtk files. Computing unprocessed frames')
        cn[:,:contact_number.shape[1]] = contact_number
        lpo[:,:local_polar_order.shape[1]] = local_polar_order 
        lno[:,:local_nematic_order.shape[1]] = local_nematic_order 

        # Compute for unprocessed frames
        for jframe in np.arange(contact_number.shape[1],DSeries.nframe_):

            print('Frame = {0}/{1}'.format(jframe+1, DSeries.nframe_) )

            # Min Dist
            MD = minDistBetweenAllFilaments( 
                    DSeries.pos_minus_[:,:,jframe], 
                    DSeries.pos_plus_[:,:,jframe], 
                    DSeries.pos_minus_[:,:,jframe], 
                    DSeries.pos_plus_[:,:,jframe], 
                    DSeries.config_['box_size']) / DSeries.config_['diameter_fil']

            # Contact Number 
            cn[:,jframe] = np.sum( np.exp(-1*(MD**2)), axis=1)

            # Local Order
            lpo[:,jframe], lno[:,jframe] = calc_local_order_frame( 
                    DSeries.orientation_[:,:,jframe], 
                    MD)

        # Save
        with open(simpath / 'contact_number.npy', 'wb') as f:
            np.save(f, cn)

        # Local Order
        with open(simpath / 'local_order.npy', 'wb') as f:
            np.save(f, lpo)
            np.save(f, lno)

    if np.any(np.isnan(cn.flatten())):
        print('Some vals are still nans')

    DSeries.contact_number = cn
    DSeries.local_polar_order = lpo
    DSeries.local_nematic_order = lno
    return DSeries
# }}}
