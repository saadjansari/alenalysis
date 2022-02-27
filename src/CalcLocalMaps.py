import numpy as np
import random, pdb, os
from matplotlib import pyplot as plt
import scipy.spatial.ckdtree
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import cdist, squareform
import time
from mpl_toolkits.mplot3d import Axes3D
import shutil
from pathlib import Path
import mpmath
from numba import njit
import src.decorators

# Plotting
# PlotPackingFraction {{{
def PlotPackingFraction(FData, savepath, N=20, **kwargs):
    """ Plot the Packing Fraction for the last N frames"""
    
    print('Packing Fraction Local Map...') 

    # Get last N frames
    frames = np.arange(FData.nframe_-N,FData.nframe_)

    # Initialize map
    densityMap = LocalMap(FData.config_,rad_scaling=2.0)

    # Packing Fraction
    # Initialize density
    rho = 0*densityMap.ComputePF_v2(FData.pos_minus_[:,:,-1], FData.pos_plus_[:,:,-1])
    # rho = 0*densityMap.ComputePF(FData.pos_minus_[:,:,-1], FData.pos_plus_[:,:,-1])
    rho = np.zeros( ( [rho.shape[0], rho.shape[1], rho.shape[2], N]) )

    # Loop over frames
    for idx,jframe in enumerate(frames):
        rho[:,:,:,idx] = densityMap.ComputePF_v2(FData.pos_minus_[:,:,jframe], FData.pos_plus_[:,:,jframe])
    rho_mu = np.mean(rho, axis=-1)
    rho_med = np.median(rho, axis=-1)

    # Plot
    # densityMap.PlotTest(rho_mu, 'Packing Fraction', savepath, **kwargs)
    # pdb.set_trace()
    densityMap.PlotTest(rho_mu, 'Packing Fraction', savepath, vmin=0.0, vmax=0.40, **kwargs)
# }}} 

# PlotNumberDensity {{{
def PlotNumberDensity(FData, savepath, N=20, **kwargs):
    """ Plot the Number density for the last N frames"""
    
    print('Number density Local Map...') 

    # Get last N frames
    frames = np.arange(FData.nframe_-N,FData.nframe_)

    # Initialize map
    densityMap = LocalMap(FData.config_,rad_scaling=2.0)

    # Number Density
    # Initialize density
    nd = 0*densityMap.ComputeND(FData.pos_minus_[:,:,-1], FData.pos_plus_[:,:,-1])
    nd = np.zeros( ( [nd.shape[0], nd.shape[1], nd.shape[2], N]) )

    # Loop over frames
    for idx,jframe in enumerate(frames):
        nd[:,:,:,idx] = densityMap.ComputeND(FData.pos_minus_[:,:,jframe], FData.pos_plus_[:,:,jframe])
    nd_mu = np.mean(nd, axis=-1)

    # Plot
    densityMap.PlotTest(nd_mu, 'Number Density', savepath, **kwargs)
# }}}

# PlotNumberDensityXlink {{{
def PlotNumberDensityXlink(FData, XData, savepath, N=10, **kwargs):
    """ Plot the Number density for crosslinkers for the last N frames"""
    
    print('Number density xlink Local Map...') 

    # Get last N frames
    frames = np.arange(FData.nframe_-N,FData.nframe_)

    # Initialize map
    densityMap = LocalMap(FData.config_,rad_scaling=2.0)

    # Number Density
    # Initialize density
    nd = 0*densityMap.ComputeND(XData.pos_minus_[:,:,-1], XData.pos_plus_[:,:,-1])
    nd = np.zeros( ( [nd.shape[0], nd.shape[1], nd.shape[2], N]) )

    # Loop over frames
    for idx,jframe in enumerate(frames):
        nd[:,:,:,idx] = densityMap.ComputeND(XData.pos_minus_[:,:,jframe], XData.pos_plus_[:,:,jframe])
    nd_mu = np.mean(nd, axis=-1)

    # Plot
    densityMap.PlotTest(nd_mu, 'Number Density', savepath, **kwargs)
# }}}

# PlotSOrder {{{
def PlotSOrder(FData, savepath, N=5, **kwargs):
    """ Plot the Nematic Order for the last N frames"""
    
    print('Nematic Order Local Map...') 

    # Get last N frames
    frames = np.arange(FData.nframe_-N,FData.nframe_)

    # Initialize map
    densityMap = LocalMap(FData.config_)

    # Nematic Order
    Sdensity = 0*densityMap.ComputeSorder(FData.pos_minus_[:,:,-1], FData.pos_plus_[:,:,-1], FData.orientation_[:,:,-1])

    # Loop over frames
    for jframe in frames:
        Sdensity += densityMap.ComputeSorder(FData.pos_minus_[:,:,jframe], FData.pos_plus_[:,:,jframe], FData.orientation_[:,:,jframe])
    Sdensity = Sdensity/len(frames)

    # Plot
    densityMap.PlotTest(Sdensity, 'Nematic Order', savepath, vmin=0, vmax=1,**kwargs)
# }}} 

# PlotPOrder {{{
def PlotPOrder(FData, savepath, N=5, **kwargs):
    """ Plot the Polar Order for the last N frames"""
    
    print('Polar Order Local Map...') 

    # Get last N frames
    frames = np.arange(FData.nframe_-N,FData.nframe_)

    # Initialize map
    densityMap = LocalMap(FData.config_)

    # Polar Order
    Pdensity = 0*densityMap.ComputePorder(FData.pos_minus_[:,:,-1], FData.pos_plus_[:,:,-1], FData.orientation_[:,:,-1])

    # Loop over frames
    for jframe in frames:
        Pdensity += densityMap.ComputePorder(FData.pos_minus_[:,:,jframe], FData.pos_plus_[:,:,jframe], FData.orientation_[:,:,jframe])
    Pdensity = Pdensity/len(frames)

    # Plot
    densityMap.PlotTest(Pdensity, 'Polar Order', savepath, vmin=-1, vmax=1,**kwargs)
# }}} 

# PlotFlux{{{
def PlotFlux(FData, savepath, N=5, **kwargs):
    """ Plot the Flux for the last N frames"""
    
    print('Flux Local Map...') 

    # Get last N frames
    frames = np.arange(FData.nframe_-N,FData.nframe_)

    # Initialize map
    densityMap = LocalMap(FData.config_, rad_scaling=2.0)

    # Polar Order
    Flux = 0*densityMap.ComputeFlux(FData.pos_minus_[:,:,-1], FData.pos_plus_[:,:,-1], FData.orientation_[:,:,-1])

    # Loop over frames
    for jframe in frames:
        Flux += densityMap.ComputeFlux(FData.pos_minus_[:,:,jframe], FData.pos_plus_[:,:,jframe], FData.orientation_[:,:,jframe])
    Flux = Flux/len(frames)

    # Plot
    densityMap.PlotTest(Flux, 'Flux', savepath, **kwargs)
# }}} 

# PlotOrientation{{{
def PlotOrientation(FData, savepath, N=1, **kwargs):
    """ Plot the Orientation for the last N frames"""
    
    print('Orientation Local Map...') 

    # Get last N frames
    frames = np.arange(FData.nframe_-N,FData.nframe_)

    # Initialize map
    densityMap = LocalMap(FData.config_, rad_scaling=2.0)

    # orientation
    ort = 0*densityMap.ComputeOrientation(FData.pos_minus_[:,:,-1], FData.pos_plus_[:,:,-1], FData.orientation_[:,:,-1])
    ort = 0*densityMap.ComputeSdirector(FData.pos_minus_[:,:,-1], FData.pos_plus_[:,:,-1], FData.orientation_[:,:,-1])
    pdb.set_trace()

    # Loop over frames
    for jframe in frames:
        ort += densityMap.ComputeOrientation(FData.pos_minus_[:,:,jframe], FData.pos_plus_[:,:,jframe], FData.orientation_[:,:,jframe])
    ort = ort/len(frames)

    # Plot
    densityMap.PlotOrientation(ort, savepath, **kwargs)
# }}} 

# PlotNematicDirector {{{
def PlotNematicDirector(FData, savepath, N=1, **kwargs):
    """ Plot the Nematic Director for the last N frames"""
    
    print('Nematic Director Local Map...') 

    # Get last N frames
    frames = np.arange(FData.nframe_-N,FData.nframe_)

    # Initialize map
    densityMap = LocalMap(FData.config_, rad_scaling=3.0)

    # orientation
    ort = 0*densityMap.ComputeSdirector(FData.pos_minus_[:,:,-1], FData.pos_plus_[:,:,-1], FData.orientation_[:,:,-1])

    # Loop over frames
    for jframe in frames:
        ort += densityMap.ComputeSdirector(FData.pos_minus_[:,:,jframe], FData.pos_plus_[:,:,jframe], FData.orientation_[:,:,jframe])
    ort = ort/len(frames)

    # Plot
    densityMap.PlotNematicDirector(ort, savepath, **kwargs)
# }}} 

# LocalMap {{{
class LocalMap():
    # Class for computing local spatial maps in a variety of geometries.
    # Confined geometries (planar, spherical, and cylindrical) are also allowed
    # unconfined geometry is also allowed
    def __init__( self, config, rad_scaling=1.0):
        self.confinement_ = config['geometry']['type'] 
        self.geometry_ = config['geometry']
        self.box_size_ = config['box_size']
        self.diameter_ = config['diameter_fil']
        # Search radius
        self.search_radius_ = rad_scaling*self.diameter_

        # Build LUT based on geometry
        self.BuildLUT()
        self.LUT_ = None

        # Special region {{{
        self.zoom_cyl = True
        if self.confinement_ == 'cylindrical':
            # C1
            if self.geometry_['radius']/(5*self.diameter_) < 2:
                self.z_low = 19.75
                self.z_high = 21.75
            # C2
            elif self.geometry_['radius']/(5*self.diameter_) < 3:
                self.z_low = 9.15 
                self.z_high = 11.15
            # C3
            elif self.geometry_['radius']/(5*self.diameter_) > 3:
                self.z_low = -0.2
                self.z_high = 1.8 
        elif self.confinement_ == 'planar':
            # P1
            if np.diff(self.geometry_['range'])[0]/(10*self.diameter_) < 2:
                self.x_low = 1.1
                self.x_high = 3.1
                self.y_low = 0.6
                self.y_high = 2.6
            # P2
            elif np.diff(self.geometry_['range'])[0]/(10*self.diameter_) < 3:
                self.x_low = -0.1 
                self.x_high = 1.9
                self.y_low = -0.1
                self.y_high = 1.9
            # P3
            elif np.diff(self.geometry_['range'])[0]/(10*self.diameter_) > 3:
                self.x_low = 0.4 
                self.x_high = 2.4
                self.y_low = -0.6
                self.y_high = 1.4
        # }}}

    # Compute Packing Fraction {{{
    @src.decorators.timer
    def ComputePF( self, pos_minus, pos_plus):
        """
        Calculate the packing fraction for single time frame
        """

        # search radius
        rad = self.get_search_radius_()

        # sample points
        xv,yv,zv = self.GetSamplePoints()
        pos = np.vstack((xv.flatten(),yv.flatten(), zv.flatten()))
        pos = self.apply_PBC(pos)
        pos_tup = tuple(pos.transpose())

        # Get Sample Positions and Volumes, idx_in_original
        pos_sample, vol_sample,_ = self.SampleFilamentLength(pos_plus, pos_minus)
        pos_sample = self.apply_PBC(pos_sample)

        # Find search volume for all sample points (full sphere, or sphere intersection with geometry)
        vol_search = self.get_search_volume(pos,rad).reshape(xv.shape)

        # Initialize cKDtree
        bs = self.get_special_boxsize()
        kdtree = scipy.spatial.cKDTree( pos_sample.transpose(), boxsize=self.get_special_boxsize())
       
        # INitiliaze packing fraction
        pf = np.zeros_like(xv)

        # # Put pos_c inside periodic box
        # neighbors_all = kdtree.query_ball_point( pos_tup, r=rad)

        # 

        # Loop over sample points
        for jx in np.arange(xv.shape[0]):
            for jy in np.arange(xv.shape[1]):
                for jz in np.arange(xv.shape[2]):

                    pos_c = np.array([xv[jx,jy,jz],yv[jx,jy,jz],zv[jx,jy,jz]])
                    pos_c = self.apply_PBC(pos_c.reshape(-1,1))[:,0]

                    # # Put pos_c inside periodic box
                    neighbors = kdtree.query_ball_point( pos_c, r=rad)
                    # neighbors = neighbors_all
                    
                    # find distance to neighbors
                    if len(neighbors)>0:
                        dd0 = np.linalg.norm(pos_c.reshape(-1,1)-pos_sample[:,neighbors], axis=0)
                        dd1 = cdist_pbc(pos_c, pos_sample[:,neighbors].T, self.get_special_boxsize())
                        # if np.max( np.abs(dd0-dd1)) > 0.0001:
                            # pdb.set_trace()
                        vol_scaling = 1-np.clip( ((dd1+self.get_diameter_()/2)-rad )/self.get_diameter_(), 0, 1).reshape(-1)
                        # Calculate volume occupied by neighbors
                        vol_neighbors = 0
                        for idx,neigh in enumerate(neighbors):
                            vol_neighbors += vol_scaling[idx]*vol_sample[neigh]

                        # Packing fraction
                        if vol_search[jx,jy,jz] > 0:
                            pf[jx,jy,jz] = vol_neighbors/vol_search[jx,jy,jz]
        return pf
    # }}}

    # Compute Packing Fraction v2 {{{
    @src.decorators.timer
    def ComputePF_v2( self, pos_minus, pos_plus):
        """
        Calculate the packing fraction for single time frame
        """

        # search radius
        rad = self.get_search_radius_()

        # sample points
        xv,yv,zv = self.GetSamplePoints()
        pos = np.vstack((xv.flatten(),yv.flatten(), zv.flatten()))
        pos = self.apply_PBC(pos)
        pos_tup = tuple(pos.transpose())

        # INitiliaze packing fraction
        pf = np.zeros_like(xv)

        # Get Sample Positions and Volumes, idx_in_original
        pos_sample, vol_sample,_ = self.SampleFilamentLength(pos_plus, pos_minus)
        pos_sample = self.apply_PBC(pos_sample)

        # Find search volume for all sample points (full sphere, or sphere intersection with geometry)
        vol_search = self.get_search_volume(pos,rad).reshape(xv.shape)

        # Initialize cKDtree
        bs = self.get_special_boxsize()
        kdtree = scipy.spatial.cKDTree( pos_sample.transpose(), boxsize=self.get_special_boxsize())
        # kdtree_grid = scipy.spatial.cKDTree( pos.transpose(), boxsize=self.get_special_boxsize())
        # dists = kdtree_grid.sparse_distance_matrix(kdtree, rad)

        # Put pos_c inside periodic box
        neighbors_all = kdtree.query_ball_point( pos_tup, r=rad)

        # Loop over sample points
        xv,yv,zv = np.meshgrid(np.arange(xv.shape[0]),
                np.arange(yv.shape[1]),
                np.arange(zv.shape[2]),indexing='ij')
        xv = xv.flatten()
        yv = yv.flatten()
        zv = zv.flatten()

        for idx in np.arange(pos.shape[1]):

            neighbors = neighbors_all[idx]
            if len(neighbors) == 0:
                continue

            jx = xv[idx]
            jy = yv[idx]
            jz = zv[idx]

            # find distance to neighbors
            dd1 = cdist_pbc(pos[:,idx], pos_sample[:,neighbors].T, self.get_special_boxsize())
            vol_scaling = 1-np.clip( ((dd1+self.get_diameter_()/2)-rad )/self.get_diameter_(), 0, 1).reshape(-1)
            # vol_scaling = np.array(vol_scaling[0,:]).reshape(-1)

            # Calculate volume occupied by neighbors
            vol_neighbors = np.sum(vol_scaling*vol_sample[neighbors])

            # Packing fraction
            if vol_search[jx,jy,jz] > 0:
                pf[jx,jy,jz] = vol_neighbors/vol_search[jx,jy,jz]
        pf = self.mask_geometry(pf)
        return pf
    # }}}

    # Number Density {{{
    @src.decorators.timer
    def ComputeND( self, pos_minus, pos_plus):
        """
        Calculate the number density for single time frame
        """

        # search radius
        rad = self.get_search_radius_()

        # sample points
        xv,yv,zv = self.GetSamplePoints()
        pos = np.vstack((xv.flatten(),yv.flatten(), zv.flatten()))
        pos = self.apply_PBC(pos)
        pos_tup = tuple(pos.transpose())

        # INitiliaze ND 
        nd = np.zeros_like(xv)

        # Get Sample Positions and Volumes, idx_in_original
        pos_sample, _,idx_sample = self.SampleFilamentLength(pos_plus, pos_minus)
        pos_sample = self.apply_PBC(pos_sample)

        # Find search volume for all sample points (full sphere, or sphere intersection with geometry)
        vol_search = self.get_search_volume(pos,rad).reshape(xv.shape)

        # Initialize cKDtrees
        bs = self.get_special_boxsize()
        kdtree = scipy.spatial.cKDTree( pos_sample.transpose(), boxsize=self.get_special_boxsize())

        # Put pos_c inside periodic box
        neighbors_all = kdtree.query_ball_point( pos_tup, r=rad)

        # Loop over sample points
        xv,yv,zv = np.meshgrid(np.arange(xv.shape[0]),
                np.arange(yv.shape[1]),
                np.arange(zv.shape[2]),indexing='ij')
        xv = xv.flatten()
        yv = yv.flatten()
        zv = zv.flatten()

        for idx in np.arange(pos.shape[1]):

            neighbors = neighbors_all[idx]
            jx = xv[idx]
            jy = yv[idx]
            jz = zv[idx]

            # NUmber density 
            if len(neighbors) > 0 and vol_search[jx,jy,jz] > 0:
                nd[jx,jy,jz] = np.unique( idx_sample[neighbors]).size/vol_search[jx,jy,jz]

        nd = self.mask_geometry(nd)
        return nd
    # }}}

    # Compute Nematic Order {{{
    def ComputeSorder( self, pos_minus, pos_plus, orient):
        """
        Calculate the nematic order for single time frame
        """

        # search radius
        rad = self.search_radius_

        # sample points
        xv,yv,zv = self.GetSamplePoints()
        pos = np.vstack((xv.flatten(),yv.flatten(), zv.flatten()))

        # Get Sample Positions and Volumes
        pos_sample,_,_ = self.SampleFilamentLength(pos_plus, pos_minus)
        pos_sample = self.apply_PBC(pos_sample)

        # Expand orient array
        orient_sample = np.repeat( orient, pos_sample.shape[1]/pos_plus.shape[1], axis=1)

        # Initialize cKDtree
        kdtree = scipy.spatial.cKDTree( pos_sample.transpose(), boxsize=self.get_special_boxsize())
       
        # INitiliaze nematic order 
        S = np.zeros_like(xv)

        # Loop over sample points
        for jx in np.arange(xv.shape[0]):
            for jy in np.arange(xv.shape[1]):
                for jz in np.arange(xv.shape[2]):

                    pos_c = np.array([xv[jx,jy,jz],yv[jx,jy,jz],zv[jx,jy,jz]])
                    neighbors = kdtree.query_ball_point( pos_c, r=rad)

                    # Calculate nematic order of neighbors
                    if len(neighbors) > 0:
                        sub_ort_array = orient_sample[:,neighbors]

                        # nematic order
                        S[jx,jy,jz] = calc_nematic_order(sub_ort_array)

        return S
    # }}}

    # Compute Polar Order {{{
    def ComputePorder( self, pos_minus, pos_plus, orient):
        """
        Calculate the polar order for single time frame
        """

        # search radius
        rad = self.search_radius_

        # sample points
        xv,yv,zv = self.GetSamplePoints()
        pos = np.vstack((xv.flatten(),yv.flatten(), zv.flatten()))

        # Get Sample Positions and Volumes
        pos_sample,_,_ = self.SampleFilamentLength(pos_plus, pos_minus)
        pos_sample = self.apply_PBC(pos_sample)

        # Expand orient array
        orient_sample = np.repeat( orient, pos_sample.shape[1]/pos_plus.shape[1], axis=1)

        # Initialize cKDtree
        kdtree = scipy.spatial.cKDTree( pos_sample.transpose(), boxsize=self.get_special_boxsize())
       
        # INitiliaze polar order 
        P = np.zeros_like(xv)

        # Loop over sample points
        for jx in np.arange(xv.shape[0]):
            for jy in np.arange(xv.shape[1]):
                for jz in np.arange(xv.shape[2]):

                    pos_c = np.array([xv[jx,jy,jz],yv[jx,jy,jz],zv[jx,jy,jz]])
                    neighbors = kdtree.query_ball_point( pos_c, r=rad)

                    # Calculate nematic order of neighbors
                    sub_ort_array = orient_sample[:,neighbors]

                    # polar order
                    if len(neighbors) > 0:
                        P[jx,jy,jz] = calc_polar_order(sub_ort_array)

        return P
    # }}}

    # Compute Local Orientation {{{
    def ComputeOrientation( self, pos_minus, pos_plus, orient):
        """
        Calculate the local orientation for single time frame
        """

        # search radius
        rad = self.search_radius_

        # sample points
        xv,yv,zv = self.GetSamplePoints()
        pos = np.vstack((xv.flatten(),yv.flatten(), zv.flatten()))

        # Get Sample Positions and Volumes
        pos_sample,_,_ = self.SampleFilamentLength(pos_plus, pos_minus)
        pos_sample = self.apply_PBC(pos_sample)

        # Expand orient array
        orient_sample = np.repeat( orient, pos_sample.shape[1]/pos_plus.shape[1], axis=1)

        # Initialize cKDtree
        kdtree = scipy.spatial.cKDTree( pos_sample.transpose(), boxsize=self.get_special_boxsize())
       
        # Initiliaze 
        ort = np.zeros((xv.shape[0], xv.shape[1], xv.shape[2], 3))

        # Loop over sample points
        for jx in np.arange(xv.shape[0]):
            for jy in np.arange(xv.shape[1]):
                for jz in np.arange(xv.shape[2]):

                    pos_c = np.array([xv[jx,jy,jz],yv[jx,jy,jz],zv[jx,jy,jz]])
                    neighbors = kdtree.query_ball_point( pos_c, r=rad)

                    # Calculate nematic order of neighbors
                    if len(neighbors) > 8:
                        sub_ort_array = orient_sample[:,neighbors]

                        # orientation
                        mean_ort = np.mean(sub_ort_array,axis=1)
                        mean_ort = mean_ort / np.linalg.norm(mean_ort)
                        # costheta_x = np.cos(np.arctan2(mean_ort[1], mean_ort[0]))
                        # cosphi_z = mean_ort[-1]
                        # ort[jx,jy,jz,:] = [costheta_x,cosphi_z]
                        ort[jx,jy,jz,:] = mean_ort

                    else:
                        ort[jx,jy,jz,:] = [np.nan,np.nan, np.nan]

        return ort
    # }}}

    # Compute Nematic Director {{{
    def ComputeSdirector( self, pos_minus, pos_plus, orient):
        """
        Calculate the nematic director for single time frame
        """

        # search radius
        rad = self.search_radius_

        # sample points
        xv,yv,zv = self.GetSamplePoints()
        pos = np.vstack((xv.flatten(),yv.flatten(), zv.flatten()))

        # Get Sample Positions and Volumes
        pos_sample,_,_ = self.SampleFilamentLength(pos_plus, pos_minus)
        pos_sample = self.apply_PBC(pos_sample)

        # Expand orient array
        orient_sample = np.repeat( orient, pos_sample.shape[1]/pos_plus.shape[1], axis=1)

        # Initialize cKDtree
        kdtree = scipy.spatial.cKDTree( pos_sample.transpose(), boxsize=self.get_special_boxsize())
       
        # INitiliaze nematic 
        N = np.zeros( (xv.shape[0], xv.shape[1], xv.shape[2], 3) )

        # Loop over sample points
        for jx in np.arange(xv.shape[0]):
            for jy in np.arange(xv.shape[1]):
                for jz in np.arange(xv.shape[2]):

                    pos_c = np.array([xv[jx,jy,jz],yv[jx,jy,jz],zv[jx,jy,jz]])
                    neighbors = kdtree.query_ball_point( pos_c, r=rad)

                    # Calculate nematic order of neighbors
                    if len(neighbors) > 8:
                        sub_ort_array = orient_sample[:,neighbors]

                        # nematic order
                        N[jx,jy,jz,:] = calc_nematic_director(sub_ort_array)

        return N
    # }}}

    # Compute Flux {{{
    def ComputeFlux( self, pos_minus, pos_plus, orient):
        """
        Calculate the flux for single time frame
        """

        # search radius
        rad = self.search_radius_

        # sample points
        xv,yv,zv = self.GetSamplePoints()
        pos = np.vstack((xv.flatten(),yv.flatten(), zv.flatten()))

        # Get Sample Positions and Volumes
        pos_sample,_, idx_in_original = self.SampleFilamentLength(pos_plus, pos_minus)
        pos_sample = self.apply_PBC(pos_sample)

        # Expand orient array
        orient_sample = np.repeat( orient, pos_sample.shape[1]/pos_plus.shape[1], axis=1)

        # Initialize cKDtree
        kdtree = scipy.spatial.cKDTree( pos_sample.transpose(), boxsize=self.get_special_boxsize())
       
        # INitiliaze flux
        flux = np.zeros_like(xv)

        # Loop over sample points
        for jx in np.arange(xv.shape[0]):
            for jy in np.arange(xv.shape[1]):
                for jz in np.arange(xv.shape[2]):

                    pos_c = np.array([xv[jx,jy,jz],yv[jx,jy,jz],zv[jx,jy,jz]])
                    pos_c_2d = pos_c.reshape(3,-1)
                    neighbors = kdtree.query_ball_point( pos_c, r=rad)

                    if len(neighbors) > 3:
                        # Positions and Orientations
                        sub_pos_array = pos_sample[:,neighbors]
                        sub_ort_array = orient_sample[:,neighbors]
                        flux_c = np.sum(sub_ort_array*(sub_pos_array-pos_c_2d) / np.linalg.norm(sub_pos_array-pos_c_2d, axis=0), axis=0)

                        # get unique neighbors (using original mt idx)
                        unq_ngbs, unq_idx = np.unique(idx_in_original[neighbors], return_index=True)
                        
                        # Get total flux of unique elements
                        flux[jx,jy,jz] = np.mean( flux_c[unq_idx])

        return flux
    # }}}

    # SampleFilamentLength {{{
    def SampleFilamentLength(self, pos_plus, pos_minus):

        # diameter
        dd = self.get_diameter_()
        # Filament Lengths
        fil_len = np.linalg.norm(pos_plus - pos_minus, axis=0)
        fil_len_max = np.max(fil_len)
        
        # number of sampling points
        n_samples = 1*int(np.ceil(fil_len_max / dd) )

        # length of sample point for each filament
        len_sample_filament = fil_len / n_samples

        # volume of sample point for each filament (bulk, and end)
        vol_sample_filament_bulk = np.pi*len_sample_filament*(dd/2)**2
        vol_sample_filament_end = np.pi*(len_sample_filament/2)*(dd/2)**2 + (4/6)*np.pi*(dd/2)**3

        # Break each filament into cylinders with diameter and height equal to the filament diameter
        pos_sampled = np.linspace(pos_minus, pos_plus, n_samples, axis=1).reshape((pos_minus.shape[0], -1), order='F')
        vol_sampled = np.repeat(vol_sample_filament_bulk,n_samples)
        vol_sampled[0::n_samples] = vol_sample_filament_end
        vol_sampled[n_samples-1::n_samples] = vol_sample_filament_end

        idx_in_original = np.repeat(np.arange(pos_plus.shape[1]),n_samples)

        return pos_sampled, vol_sampled, idx_in_original
    # }}}

    # Sample Points {{{
    def GetSamplePoints( self):
        # Get 3D sample points based on geometry
        # Points will sample a 3D rectangular region for each geometry

        # Point spacing
        width = self.get_diameter_()

        # Get lower and upper bound for rectangular region
        if self.confinement_ == 'unconfined':
            bs_low = np.array([0,0,0], dtype=np.float32)
            bs_high = self.get_box_size_()
            
        if self.confinement_ == 'planar':
            zrange = np.copy( np.array( self.geometry_['range'] ) )
            bs_low = np.array([0,0,0], dtype=np.float32)
            bs_high = self.get_box_size_()
            bs_low[-1] = zrange[0]
            bs_high[-1] = zrange[1]
            if self.zoom_cyl:
                bs_low[0] = self.x_low
                bs_high[0] = self.x_high
                bs_low[1] = self.y_low
                bs_high[1] = self.y_high

        if self.confinement_ == 'cylindrical':
            center = np.array(self.geometry_['center'])
            radius = self.geometry_['radius']
            bs = self.get_box_size_()
            bs_low = np.array(center-0.9*radius)-10**-6
            bs_high = np.array(center+0.9*radius)+10**-6

            d_unitless = int(2*radius/width)
            if self.zoom_cyl:
                bs_low[-1] = self.z_low
                bs_high[-1] = self.z_high
            else:
                bs_low[-1] = 0
                bs_high[-1] = bs[-1]
            xx = center[0] + width*(np.arange(d_unitless)-(d_unitless-1)/2)
            yy = center[1] + width*(np.arange(d_unitless)-(d_unitless-1)/2)
            zz = np.arange(bs_low[2], bs_high[2], width)
            xv,yv,zv = np.meshgrid(xx,yy,zz,indexing='ij')
            return xv,yv,zv

        if self.confinement_ == 'spherical':
            center = np.array(self.geometry_['center'])
            radius = self.geometry_['radius']
            d_unitless = int(2*radius/width)
            xx = center[0] + width*(np.arange(d_unitless)-(d_unitless-1)/2)
            yy = center[1] + width*(np.arange(d_unitless)-(d_unitless-1)/2)
            zz = center[2] + width*(np.arange(d_unitless)-(d_unitless-1)/2)
            xv,yv,zv = np.meshgrid(xx,yy,zz,indexing='ij')
            return xv,yv,zv
            # bs_low = np.array(center-0.9*radius)-10**-6
            # bs_high = np.array(center+0.9*radius)+10**-6
        
        xx = np.arange(bs_low[0], bs_high[0], width)
        yy = np.arange(bs_low[1], bs_high[1], width)
        zz = np.arange(bs_low[2], bs_high[2], width)
        
        # Apply PBC here
        bs = self.get_special_boxsize()
        xv,yv,zv = np.meshgrid(xx,yy,zz,indexing='ij')
        return xv,yv,zv
    # }}}

    # Special BoxSize {{{
    def get_special_boxsize(self):

        if self.confinement_ == 'unconfined':
            box_size_new = self.get_box_size_()
        elif self.confinement_ == 'spherical':
            box_size_new = 100*self.get_box_size_()
        elif self.confinement_ == 'cylindrical':
            cyl_height = self.get_box_size_()[2]
            box_size_new = np.array([10*cyl_height, 10*cyl_height, cyl_height])
        elif self.confinement_ == 'planar':
            bs = self.get_box_size_()
            box_size_new = np.array([
                bs[0], bs[1], 10*np.max(bs)
                ])
        return box_size_new
    # }}}

    # Get Search Volume {{{
    def get_search_volume(self,pos,r):
        # Get search Volume based on point positions, and geometry and 
        # pos: coorindates 3 x N (where N is number of filaments )
        # existence of LUTs
        
        vol_search = np.zeros((1, pos.shape[-1]))
        if self.confinement_ == 'unconfined':
            vol_search[:] = self.vol_sphere_unconfined(r)

        elif self.confinement_ == 'cylindrical':
            cyl_center = np.array( self.geometry_['center'] )
            cyl_radius = self.geometry_['radius'] 

            # impact parameter b
            # This is minimum distance from cylinder axis to point
            b_vals = np.linalg.norm( pos[0:2,:]-cyl_center[0:2].reshape(-1,1), 
                    axis=0)

            if self.LUT_ is None:
                self.BuildLUT()
            vol_search = self.LUT_(b_vals) 

        elif self.confinement_ == 'spherical':
            # Calculate distance from Confinement sphere center for all pts
            sph_center = np.array( self.geometry_['center'] )
            sph_radius = self.geometry_['radius'] 
            d_vals = np.linalg.norm( pos-sph_center.reshape(-1,1), axis=0)
            if self.LUT_ is None:
                self.BuildLUT()
            vol_search = self.LUT_(d_vals)

        elif self.confinement_ == 'planar':

            zrange = np.array( self.geometry_['range'] )
            z_vals = pos[-1,:]

            if self.LUT_ is None:
                self.BuildLUT()
            vol_search = self.LUT_(z_vals)
        return vol_search
    # }}}

    # Volume Formulas {{{
    # Volume Confined Cylindrical Sphere/Shell {{{
    @staticmethod
    def vol_sphere_confined_cylindrical(R,r,b):
        """
        Adapted from
        Lamarche, F., & Leroy, C. (1990).
        Evaluation of the volume of intersection of a sphere
        with a cylinder by elliptic integrals.
        Computer Physics Communications, 59(2), 359-369.

        R: cylinder radius
        r: sphere radius
        b: minimum distance from axis of cylinder to center of sphere.
        """
        vol=0
        eps=1e-7

        # impact parameter
        b += eps

        # useful quantities
        AA = np.max( [r**2, (b+R)**2])
        BB = np.min( [r**2, (b+R)**2])
        CC = (b-R)**2
        ss = (b+R)*(b-R)

        if r < R-b:
            return (4*np.pi/3)*(r**3)
        elif r == b+R:
            t1 = (4*np.pi/3)*(r**3)*np.heaviside(R-b,0.5)
            t2 = (4/3)*(r**3)*np.arctan2(2*np.sqrt(b*R),b-R)
            t3 = (4/3)*np.sqrt(AA-CC)*(ss+(2/3)*(AA-CC))
            vol = t1+t2-t3
            return vol
        else:
            k2 = (BB-CC)/((AA-CC))
            na2 = (BB-CC)/(CC)
            E_K = float(mpmath.ellipk(k2))
            E_E = float(mpmath.ellipe(k2))
            E_P = float(mpmath.ellippi(-na2,-eps + np.pi/2, k2))

            if r < b+R:
                t1 = (4*np.pi/3)*(r**3)*np.heaviside(R-b,0.5)
                t_factor = 4/(3*np.sqrt(AA-CC))
                t2 = E_P*ss*(BB**2)/CC
                t3 = E_K*( ss*(AA-2*BB) + (AA-BB)*(3*BB - CC - 2*AA)/(3) )
                t4 = E_E*(AA-CC)*(-ss+(2*AA + 2*CC - 4*BB)/3)
                vol = t1 + t_factor*(t2+t3+t4)

            elif r > b+R:
                t1 = (4*np.pi/3)*(r**3)*np.heaviside(R-b,0.5)
                t_factor = 4/(3*np.sqrt(AA-CC))
                t2 = E_P*ss*(AA**2)/CC
                t3 = E_K*(AA*ss - (AA-BB)*(AA-CC)/3)
                t4 = E_E*(AA-CC)*(ss + (4*AA - 2*BB - 2*CC)/3)
                vol = t1 + t_factor*(t2-t3-t4)

            else:
                raise Exception('one condition must be met!')
        return vol
    # }}}

    # Volume Confined Spherical Sphere/Shell {{{
    @staticmethod
    def vol_sphere_confined_spherical(R,r,d):
        """
        R: confining sphere radius
        r: probe sphere radius
        d: distance between sphere centers.
        """
        if d >= R+r:
            vol = 0
        elif d <= np.absolute(R-r):
            vol = (4/3)*np.pi*np.min([r,R])**3
        else:
            vol = np.pi*((R+r-d)**2)*(d**2 + 2*d*r - 3*r**2 + 2*d*R + 6*r*R - 3*R**2) / (12*d)
        return vol
    # }}}

    # Volume Confined Planar Sphere/Shell {{{
    @staticmethod
    def vol_sphere_confined_planar(r, z_min, z_max, z):
        # calculated by subtracting caps
        vol_sphere_r = (4/3)*np.pi*(r**3)
        h0_min = z_min - (z-r)
        if h0_min < 0:
            h0_min = 0
        h0_max = z+r-z_max
        if h0_max < 0:
            h0_max = 0
        vol = vol_sphere_r - ( (np.pi/3)*(h0_min**2)*(3*r - h0_min)) - ( (np.pi/3)*(h0_max**2)*(3*r - h0_max))
        return vol
    # }}}

    # Volume Unconfined Sphere/Shell {{{
    @staticmethod
    def vol_sphere_unconfined(r):
        return (4/3)*np.pi*(r**3)
    # }}}
    # }}}
    
    # LUT {{{
    def BuildLUT(self):
        # Build Look up table if required
        # speeds up computation for confinement geometries
        
        radii = self.search_radius_

        if self.confinement_ == 'cylindrical':

            R = self.geometry_['radius']
            b_vals = np.linspace(0.0,1.1*R,100)
            self.LUT_ = self.LUT_cylindrical(R,radii,b_vals)

        elif self.confinement_ == 'spherical':

            R = self.geometry_['radius']
            d_vals = np.linspace(0.0,1.1*R,100)
            self.LUT_ = self.LUT_spherical(R,radii,d_vals)

        elif self.confinement_ == 'planar':

            zrange = self.geometry_['range']
            z_vals = np.linspace(zrange[0],zrange[1],50)
            self.LUT_ = self.LUT_planar(zrange,radii,z_vals)

    @staticmethod
    def LUT_cylindrical(R,rad,b_vals):

        print('Building LUT...')
        data = np.zeros(len(b_vals))

        count = 0
        for idx_b,bb in enumerate(b_vals):
            data[idx_b] = LocalMap.vol_sphere_confined_cylindrical(R,rad,bb)

            count+=1
            print('Progress = {0:.0f}%'.format(100*count/len(b_vals)), end='\r',flush=True)

        print('LUT built successfully!')
        return UnivariateSpline(b_vals,data)

    @staticmethod
    def LUT_spherical(R,rad,d_vals):

        print('Building LUT...')
        data = np.zeros( len(d_vals))

        count = 0
        for idx_d,dd in enumerate(d_vals):
            data[idx_d] = LocalMap.vol_sphere_confined_spherical(R,rad,dd)

            count+=1
            print('Progress = {0:.0f}%'.format(100*count/len(d_vals)), end='\r',flush=True)

        print('LUT built successfully!')
        return UnivariateSpline(d_vals,data)

    @staticmethod
    def LUT_planar(zrange,rad,z_vals):

        print('Building LUT...')
        data = np.zeros( len(z_vals))

        count = 0
        for idx_z,zz in enumerate(z_vals):
            data[idx_z] = LocalMap.vol_sphere_confined_planar(rad,zrange[0],zrange[1],zz)

            count+=1
            print('Progress = {0:.0f}%'.format(100*count/len(z_vals)), end='\r',flush=True)

        print('LUT built successfully!')
        return UnivariateSpline(z_vals,data)
    # }}}

    # Apply PBC {{{
    def apply_PBC(self, pos):
        # pos: coorindates 3 x N (where N is number of pts)
        # Apply periodic BC
        # ensure points are within boxsize
        
        bs = self.get_special_boxsize()
        for jdim in range(pos.shape[0]):
            pos[jdim, pos[jdim,:] < 0] += bs[jdim]
            pos[jdim, pos[jdim,:] < 0] += bs[jdim]
            pos[jdim, pos[jdim,:] > bs[jdim]] -= bs[jdim]
            pos[jdim, pos[jdim,:] > bs[jdim]] -= bs[jdim]
        return pos
    # }}}

    # PlotTest {{{
    def PlotTest(self, data, label, savepath, **kwargs):

        # sample points
        xv,yv,zv = self.GetSamplePoints()

        # colormap
        colormap = 'RdYlBu_r'
        # colormap range
        if 'vmin' in kwargs:
            vmin = kwargs['vmin']
        else:
            vmin = np.min([ 
                np.min(np.mean(data,axis=2)),
                np.min(np.mean(data,axis=1)),
                np.min(np.mean(data,axis=0))
                ]) 
        if 'vmax' in kwargs:
            vmax = kwargs['vmax']
        else:
            vmax = np.max([ 
                np.max(np.mean(data,axis=2)),
                np.max(np.mean(data,axis=1)),
                np.max(np.mean(data,axis=0))
                ]) 
        
        # edges
        xe = np.around([np.min(xv.flatten()), np.max(xv.flatten())],2)
        ye = np.around([np.min(yv.flatten()), np.max(yv.flatten())],2)
        ze = np.around([np.min(zv.flatten()), np.max(zv.flatten())],2)

        # Dump to csv
        dat = np.vstack((xv.flatten(), yv.flatten(), zv.flatten(),data.flatten())).transpose()
        np.savetxt(savepath.parent/'pf_data.csv', dat, delimiter=',')

        # 2D histograms (XY,XZ,YZ)
        fig,axs = plt.subplots(1,3, figsize=(12,3))

        # XY
        idx0 = np.arange( int((data.shape[2]-1)/2), 3+int((data.shape[2]-1)/2))
        im0 = axs[0].imshow(np.mean(data[:,:,idx0],axis=2).transpose(), 
                cmap=colormap, interpolation='quadric', 
                origin='lower', vmin=vmin, vmax=vmax,
                extent=( xe[0], xe[1], ye[0], ye[1]) )
        cb = plt.colorbar(im0, ax=axs[0], label=label)
        axs[0].set(xlabel=r'$X (\mu m)$', ylabel=r'$Y (\mu m)$')
        axs[0].set_xticks(xe)
        axs[0].set_yticks(ye)
        
        # XZ
        idx0 = np.arange( int((data.shape[1]-1)/2), 3+int((data.shape[1]-1)/2))
        im1 = axs[1].imshow(np.mean(data[:,idx0,:],axis=1).transpose(), 
                cmap=colormap, interpolation='quadric', 
                origin='lower', vmin=vmin, vmax=vmax,
                extent=( xe[0], xe[1], ze[0], ze[1]) )
        plt.colorbar(im1, ax=axs[1], label=label)
        axs[1].set(xlabel=r'$X (\mu m)$', ylabel=r'$Z (\mu m)$')
        axs[1].set_xticks(xe)
        axs[1].set_yticks(ze)

        # YZ
        idx0 = np.arange( int((data.shape[0]-1)/2), 3+int((data.shape[0]-1)/2))
        im2 = axs[2].imshow(np.mean(data[idx0,:,:],axis=0).transpose(), 
                cmap=colormap, interpolation='quadric',
                origin='lower', vmin=vmin, vmax=vmax,
                extent=( ye[0], ye[1], ze[0], ze[1]) )
        plt.colorbar(im2, ax=axs[2], label=label, ticks=[0.0,0.2,0.4])
        axs[2].set(xlabel=r'$Y (\mu m)$', ylabel=r'$Z (\mu m)$')
        axs[2].set_xticks(ye)
        axs[2].set_yticks(ze)

        plt.tight_layout()
        plt.savefig(savepath, bbox_inches="tight")
        plt.close()
        # }}}

    # PlotOrientation {{{
    def PlotOrientation(self, data, savepath, **kwargs):


        label = ['Theta (rad)', 'Phi (rad)']
        # sample points
        xv,yv,zv = self.GetSamplePoints()

        # colormap
        colormap = ['hsv', 'RdYlBu_r']
        
        # colormap range
        vmin = [0,0]
        vmax = [2*np.pi,np.pi]
        
        # edges
        xe = np.around([np.min(xv.flatten()), np.max(xv.flatten())],2)
        ye = np.around([np.min(yv.flatten()), np.max(yv.flatten())],2)
        ze = np.around([np.min(zv.flatten()), np.max(zv.flatten())],2)

        # 2D histograms (XY,XZ,YZ)
        fig,axs = plt.subplots(2,3, figsize=(12,7))

        for jrow in range(2):
            
            vlow = vmin[jrow]
            vhigh = vmax[jrow]
            cmap = colormap[jrow]
            
            # XY
            datXY = np.nanmean(data,axis=2)
            if jrow == 0:
                angle = np.mod( np.arctan2(datXY[:,:,1], datXY[:,:,0]).transpose(), 2*np.pi)
            elif jrow == 1:
                angle = np.arccos(datXY[:,:,2]).transpose()

            im0 = axs[jrow,0].imshow(angle, 
                    cmap=cmap, interpolation='nearest', 
                    origin='lower', vmin=vlow, vmax=vhigh,
                    extent=( xe[0], xe[1], ye[0], ye[1]) )
            cb = plt.colorbar(im0, ax=axs[jrow,0], label=label[jrow])
            axs[jrow,0].set(xlabel=r'$X (\mu m)$', ylabel=r'$Y (\mu m)$')
            axs[jrow,0].set_xticks(xe)
            axs[jrow,0].set_yticks(ye)
        
            # XZ
            datXZ = np.nanmean(data,axis=1)
            if jrow == 0:
                angle = np.mod( np.arctan2(datXZ[:,:,1], datXZ[:,:,0]).transpose(), 2*np.pi)
            elif jrow == 1:
                angle = np.arccos(datXZ[:,:,2]).transpose()
            im1 = axs[jrow,1].imshow(angle, 
                    cmap=cmap, interpolation='nearest', 
                    origin='lower', vmin=vlow, vmax=vhigh,
                    extent=( xe[0], xe[1], ze[0], ze[1]) )
            plt.colorbar(im1, ax=axs[jrow,1], label=label[jrow])
            axs[jrow,1].set(xlabel=r'$X (\mu m)$', ylabel=r'$Z (\mu m)$')
            axs[jrow,1].set_xticks(xe)
            axs[jrow,1].set_yticks(ze)

            # YZ
            datYZ = np.nanmean(data,axis=0)
            if jrow == 0:
                angle = np.mod( np.arctan2(datYZ[:,:,1], datYZ[:,:,0]).transpose(), 2*np.pi)
            elif jrow == 1:
                angle = np.arccos(datYZ[:,:,2]).transpose()
            im2 = axs[jrow,2].imshow(angle, 
                    cmap=cmap, interpolation='nearest',
                    origin='lower', vmin=vlow, vmax=vhigh,
                    extent=( ye[0], ye[1], ze[0], ze[1]) )
            plt.colorbar(im2, ax=axs[jrow,2], label=label[jrow])
            axs[jrow,2].set(xlabel=r'$Y (\mu m)$', ylabel=r'$Z (\mu m)$')
            axs[jrow,2].set_xticks(ye)
            axs[jrow,2].set_yticks(ze)

        plt.tight_layout()
        plt.savefig(savepath, bbox_inches="tight")
        plt.close()
        # }}}
        
    # PlotNematicDirector {{{
    def PlotNematicDirector(self, data, savepath, **kwargs):

        # sample points
        xv,yv,zv = self.GetSamplePoints()

        ax = plt.figure(figsize=(4,8)).add_subplot(projection='3d')
        ax.quiver(xv, yv, zv, data[:,:,:,0], data[:,:,:,1], data[:,:,:,2], pivot='middle', length=0.03)
        ax.set(xlabel=r'$X (\mu m)$', ylabel=r'$Y (\mu m)$', zlabel=r'$Z (\mu m)$')
        plt.tight_layout()
        ax.set_box_aspect((1, 1, 1))
        plt.show()
        plt.savefig(savepath, bbox_inches="tight")
        plt.close()
        # }}}

    # Mask Geometry {{{
    def mask_geometry(self, data):
        
        data_masked = np.copy(data)
        xv,yv,zv = self.GetSamplePoints()

        if self.confinement_ == 'spherical':
            
            center = np.array(self.geometry_['center'])
            radius = self.geometry_['radius']

            for xj in np.arange(xv.shape[0]):
                for yj in np.arange(yv.shape[1]):
                    for zj in np.arange(zv.shape[2]):
                        
                        if np.linalg.norm([xv[xj,yj,zj], yv[xj,yj,zj], zv[xj,yj,zj]] - center) > radius+self.diameter_/10:
                            data_masked[xj,yj,zj] = 0

        elif self.confinement_ == 'cylindrical':
            
            center = np.array(self.geometry_['center'])
            radius = self.geometry_['radius']

            for xj in np.arange(xv.shape[0]):
                for yj in np.arange(yv.shape[1]):
                    if np.linalg.norm([xv[xj,yj,0], yv[xj,yj,0]]-center[:2]) > radius+self.diameter_/10:
                        data_masked[xj,yj,:] = 0

        return data_masked

    # Get {{{
    def get_box_size_(self):
        return np.copy(self.box_size_)
    def get_confinement_(self):
        return self.confinement_
    def get_geometry_(self):
        return self.geometry_
    def get_diameter_(self):
        return self.diameter_
    def get_search_radius_(self):
        return self.search_radius_
    # }}}
# }}}

# Order Params {{{
def calc_nematic_order(orient_array):
    """
    Calculates the nematic order S by computing the maximum eigenvalue
    of the nematic tensor Q
    Inputs: 
        orient_array : 3 x N (where N is number of filaments, T is number of frames)
    """

    # calculate Q tensor
    Q = calc_nematic_tensor(orient_array)

    # Find largest eigenvalue
    S = np.sqrt(np.tensordot(Q, Q)*1.5)
    return S

def calc_nematic_director(orient_array):
    """
    Calculates the nematic director N by computing the eigenvector correpsinding to the maximum eigenvalue
    of the nematic tensor Q
    Inputs: 
        orient_array : 3 x N (where N is number of filaments, T is number of frames)
    """

    # calculate Q tensor
    Q = 1.5*calc_nematic_tensor(orient_array)

    # Find largest eigenvalue
    vals, vecs = np.linalg.eig(Q)

    # Find the index of the maximum eigenvalue (absolute magnitude)
    idxMax = np.where( np.abs(vals) == np.max(np.abs(vals)))[0][0]
    
    # eigenvector
    if vals[idxMax] < 0:
        ev = -vecs[:,idxMax]
    else:
        ev = vecs[:,idxMax]
    return ev

@njit
def calc_nematic_tensor( orient_array):
    """
    Calculates the nematic tensor Q
    Inputs: 
        orient_array : 3 x N x T (where N is number of filaments, T is number of frames)
    """

    # Num filaments
    n_fil = orient_array.shape[1]

    # initialize Q tensor
    Q = np.zeros((3,3))

    # sum over all filaments, taking their outer products
    for irow in np.arange(n_fil):
        Q += np.outer( orient_array[:,irow], orient_array[:,irow])

    # complete mean calculation by dividing by number of filaments and subtracting identity.
    Q = Q/n_fil - np.identity(3)/3
    return Q

def calc_polar_order(orient_array):
    """
    Calculate the polar order
    Inputs: 
        orient_array : 3 x N (where N is number of filaments, T is number of frames)
    """

    # Num filaments
    n_fil = orient_array.shape[1]

    # Initialize P vector
    Pvec = np.zeros(3)

    # Take a mean of all orientations
    for irow in np.arange(n_fil):
        Pvec += orient_array[:,irow]
    Pvec = Pvec / n_fil
    P = np.linalg.norm( Pvec)
    return P
# }}}

# dist_pbc {{{
def cdist_pbc(pos_ref, pos, box_size):
    """ pos_ref dimension is 1 x 3 """
    """ pos is dimension N x 3 """
    for jd in range(pos.shape[1]):
        cd = cdist(pos_ref[jd].reshape(-1,1), pos[:,jd].reshape(-1,1))
        cd[ cd > 0.5*box_size[jd] ] -= box_size[jd]
        try:
            total += cd**2
        except:
            total = cd**2
    total = np.sqrt(total)
    return total
# }}}
