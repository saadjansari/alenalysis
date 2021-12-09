import numpy as np
import random, pdb, os
from matplotlib import pyplot as plt
import scipy.spatial.ckdtree
from scipy.interpolate import RectBivariateSpline
import time
from mpl_toolkits.mplot3d import Axes3D
import shutil
from pathlib import Path

# Plotting
def PlotFilamentDensityLastNframes(FData, savepath, N=10,**kwargs):
    """ Plot the RDF for the last N frames"""
    
    frames = np.arange(FData.nframe_-N,FData.nframe_)

    print('Density Maps...') 
    rhoMap = DensityMap(FData.config_, **kwargs)
    rho = 0*rhoMap.Compute(FData.pos_minus_[:,:,-1], FData.pos_plus_[:,:,-1])
    for jframe in frames:
        rho += rhoMap.Compute(FData.pos_minus_[:,:,jframe], FData.pos_plus_[:,:,jframe])
    rho = rho/len(frames)

    # Find max values for colormap
    vmax = 0
    for jdim in range(len(rho.shape)):
        vmax = np.max([vmax, np.max(np.mean(rho, axis=jdim))])
            
    fig,ax = rhoMap.Plot2D(rho, vmax=vmax, colormap='hot')
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()
    
# Plot Density movie {{{
def PlotFilamentDensityMovie(FData, savedir, start_frame=0,frame_gap=10,colormap='hot',**kwargs):
    """ Make a Density Movie for all frames"""
    
    # Paths
    if Path.exists(savedir):
        shutil.rmtree( savedir)
    os.mkdir( savedir)
        
    print('Making Filament Density Movie...') 

    # frames
    frames = np.arange(start_frame, FData.nframe_,frame_gap)

    # Initialize density map for this simulation
    rhoMap = DensityMap(FData.config_, **kwargs)
    rho = 0*rhoMap.Compute(FData.pos_minus_[:,:,-1], FData.pos_plus_[:,:,-1])

    # Find density for each frame and save in a list
    rho_all = []
    for jframe in frames:
        print('Frame = {0}'.format(jframe))
        rho_all.append( rhoMap.Compute(FData.pos_minus_[:,:,jframe], FData.pos_plus_[:,:,jframe]) )

    # Find max values for colormap
    vmax = 0
    for idx,jframe in enumerate(frames):
        for jdim in range(len(rho.shape)):
            vmax = np.max([vmax, np.max(np.mean(rho_all[idx], axis=jdim))])
        
    # Save images to folder
    for idx,jframe in enumerate(frames):
        save_pt = savedir / 'frame{:04d}.png'.format(idx)
        fig,ax = rhoMap.Plot2D(rho_all[idx], colormap=colormap, vmax=vmax)
        plt.suptitle('Frame = {0}'.format(jframe) )
        plt.tight_layout()
        plt.savefig(save_pt, bbox_inches="tight")
        plt.close()
    
    # Save movie 
    movie_path = savedir / 'density_movie.mp4'
    image_path = savedir / 'frame*.png'
    os.system("ffmpeg -framerate 20 -pattern_type glob -i '{0}' -c:v libx264 -pix_fmt yuv420p -profile:v high -level 4.2 -crf 17 -vf scale=1920:-2 -r 25 {1}".format(image_path,movie_path) )
    # }}}

class DensityMap():
    # Class for computing density maps in a variety of geometries.
    # Confined geometries (planar, spherical, and cylindrical) are also allowed
    # unconfined geometry is also allowed
    def __init__(self, config, diameter_factor=2):
        self.confinement_ = config['geometry']['type'] 
        self.geometry_ = config['geometry']
        self.box_size_ = config['box_size']
        self.diameter_ = config['diameter_fil']
        self.diameter_factor_ = diameter_factor
        self.zoom = False

    # Compute Density Map {{{
    def Compute( self, pos_minus, pos_plus):
        """
        Calculate the density map for a single time frame
        Inputs:
            pos: coorindates 3 x N (where N is number of filaments )
        """
        # Get Bins
        bins,_ = self.GetBins()

        # Get points number density
        # num_density = self.calc_number_density(pos.shape[1])

        # Get Sample Positions and Volumes
        pos_sample, vol_sample = self.SampleFilamentLength(pos_plus, pos_minus)

        # Get Occupied Volume in bins
        vol_in_bin = self.GetHistogramCount(pos_sample, vol_sample, bins)

        # Get volume of each bin
        bin_volumes = self.GetBinVolume(bins)

        # Divide occupied volume by bin volume to get a density
        density = np.divide(vol_in_bin, bin_volumes)
    
        # Divide by global density to normalize 
        global_density = self.GetTotalFilamentVolume(pos_minus, pos_plus) / self.GetGlobalVolume()
        density_normalized = density / global_density

        return density_normalized
    # }}}

    # Get Bins {{{
    def GetBins(self):
        # Get bins based on geometry
        # 1. Unconfined / Planar : 3D bins (X,Y,Z)
        # 2. Cylindrical : 2D bins (R,Z)
        # 3. Spherical : 1D bins (R)

        # Bin width
        width= self.diameter_factor_*self.diameter_

        if self.confinement_ == 'unconfined':
            bs_low = np.array([0,0,0])
            bs_high = self.box_size_
            if self.zoom:
                bs_low = np.array([1.2,1.2,1.2])
                bs_high = np.array([2.8,2.8,2.8])

            bins = [ np.arange(v_min, v_max, width) for v_min,v_max in zip(bs_low, bs_high) ]
            
        if self.confinement_ == 'planar':
            zrange = np.array( self.geometry_['range'] )
            bs_low = np.array([0,0,0])
            bs_high = self.box_size_
            bs_low[-1] = zrange[0]
            bs_high[-1] = zrange[1]
            bins = [ np.arange(v_min, v_max, width) for v_min,v_max in zip(bs_low, bs_high) ]

        if self.confinement_ == 'cylindrical':
            r_max = self.geometry_['radius']
            z_low = 0
            z_high = self.box_size_[-1]
            bins = [ np.arange(0,r_max,width), np.arange(z_low,z_high,width) ]

        if self.confinement_ == 'spherical':
            r_max = self.geometry_['radius']
            bins = [ np.arange(0,r_max,width) ]
        
        bin_centers = [ (jbins[1:]+jbins[:-1])/2 for jbins in bins]

        return bins, bin_centers
    # }}}

    # SampleFilamentLength {{{
    def SampleFilamentLength(self, pos_plus, pos_minus):

        # Filament Lengths
        fil_len = np.linalg.norm(pos_plus - pos_minus, axis=0)
        fil_len_max = np.max(fil_len)
        
        # number of sampling points
        n_samples = 1*int(np.ceil(fil_len_max / self.diameter_) )

        # length of sample point for each filament
        len_sample_filament = fil_len / n_samples

        # volume of sample point for each filament
        vol_sample_filament = np.pi*len_sample_filament*(self.diameter_/2)**2

        # Break each filament into cylinders with diameter and height equal to the filament diameter
        pos_sampled = np.linspace(pos_minus, pos_plus, n_samples, axis=1).reshape((pos_minus.shape[0], -1), order='F')
        vol_sampled = np.repeat(vol_sample_filament,n_samples)

        return pos_sampled, vol_sampled
    # }}}

    # GetHistogramCount {{{
    def GetHistogramCount(self, pos_sample, vol_sample, bins):

        # If cylindrical or spherical, transform position coordinates
        if self.confinement_ == 'cylindrical':
            pos_sample[:2,:] = pos_sample[:2,:] - np.array(self.geometry_['center'][:2]).reshape(-1,1)
            radii = np.linalg.norm(pos_sample[:2,:], axis=0)
            zvals = pos_sample[2,:]
            pos_new = np.vstack((radii,zvals))
        
        elif self.confinement_ == 'spherical':
            pos_sample = pos_sample - np.array(self.geometry_['center']).reshape(-1,1)
            radii = np.linalg.norm(pos_sample, axis=0)
            pos_new = radii

        else:
            pos_new = pos_sample

        counts = np.histogramdd( pos_new.transpose(), bins, weights=vol_sample)[0]
        return counts
    # }}}

    # Get Bin Volume {{{
    def GetBinVolume(self, bins):

        num_bins = [len(jbin)-1 for jbin in bins]
        vols = np.zeros( num_bins)

        if self.confinement_ == 'cylindrical':
            for ridx in range( num_bins[0]):
                r1 = bins[0][ridx+1] 
                r0 = bins[0][ridx] 
                for zidx in range( num_bins[1]):
                    z1 = bins[1][zidx+1]
                    z0 = bins[1][zidx]
                    vols[ridx,zidx] = np.pi*((r1**2)-(r0**2))*(z1-z0)
        elif self.confinement_ == 'spherical':
            for ridx in range( num_bins[0]):
                r1 = bins[0][ridx+1] 
                r0 = bins[0][ridx] 
                vols[ridx,zidx] = (4/3)*np.pi*((r1**3)-(r0**3))
        else:
            for xidx in range( num_bins[0]):
                x1 = bins[0][xidx+1]
                x0 = bins[0][xidx]
                for yidx in range( num_bins[1]):
                    y1 = bins[1][yidx+1]
                    y0 = bins[1][yidx]
                    for zidx in range( num_bins[2]):
                        z1 = bins[2][zidx+1]
                        z0 = bins[2][zidx]
                        vols[xidx,yidx,zidx] = (x1-x0)*(y1-y0)*(z1-z0)
        return vols
    # }}}
    
    # GetGlobalVolume {{{
    def GetGlobalVolume(self):
        # Calculate the global volume of this geometry

        if self.confinement_ == 'unconfined':
            vol = np.prod(self.box_size_)
        elif self.confinement_ == 'cylindrical':
            vol = np.pi * self.box_size_[-1] * self.geometry_['radius']**2
        elif self.confinement_ == 'spherical':
            vol = (4/3) * np.pi * self.geometry_['radius']**3
        elif self.confinement_ == 'planar':
            vol = np.diff( self.geometry_['range']) * np.prod(self.box_size_[:2])
        return vol 
    # }}}

    # GetTotalFilamentVolume {{{
    def GetTotalFilamentVolume(self, pos_minus, pos_plus):

        # Filament Lengths
        fil_len = np.linalg.norm(pos_plus - pos_minus, axis=0)

        fil_vol = np.sum( np.pi*(self.diameter_**2)*fil_len/4)
        return fil_vol
    # }}}

    # Plotting {{{
    def Plot2D(self, rho, **kwargs):

        bin_edges, bin_centers = self.GetBins()
        if self.confinement_ == 'unconfined':
            fig,ax = self.PlotUnconfined2D(rho, bin_centers, bin_edges, **kwargs)
        elif self.confinement_ == 'cylindrical':
            fig,ax = self.PlotCylindrical2D(rho, bin_centers, bin_edges, **kwargs)
        elif self.confinement_ == 'spherical':
            fig,ax = self.PlotSpherical2D(rho, bin_centers, bin_edges, **kwargs)
        elif self.confinement_ == 'planar':
            fig,ax = self.PlotPlanar2D(rho, bin_centers, bin_edges, **kwargs)
        return fig,ax

    # PlotUnconfined {{{
    def PlotUnconfined2D(self, rho, bin_centers, bin_edges, colormap='RdYlBu_r',**kwargs):

        if 'vmin' in kwargs:
            vmin = kwargs['vmin']
        else:
            vmin = 0

        if 'vmax' in kwargs:
            vmax = kwargs['vmax']
        else:
            vmax = np.max( [ np.max(np.mean(rho,axis=2)),np.max(np.mean(rho,axis=1)),np.max(np.mean(rho,axis=0)) ] ) 
        
        # edges
        xe = [bin_edges[0][0], bin_edges[0][-1]]
        ye = [bin_edges[1][0], bin_edges[1][-1]]
        ze = [bin_edges[2][0], bin_edges[2][-1]]

        # 2D histograms (XY,XZ,YZ)
        fig,axs = plt.subplots(1,3, figsize=(12,3))

        # XY
        im0 = axs[0].imshow(np.mean(rho,axis=2).transpose(), cmap=colormap, interpolation='quadric', 
                origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                extent=( xe[0], xe[1], ye[0], ye[1]) )
        cb = plt.colorbar(im0, ax=axs[0], label='Density')
        axs[0].set(xlabel=r'$X (\mu m)$', ylabel=r'$Y (\mu m)$')
        axs[0].set_xticks(xe)
        axs[0].set_yticks(ye)
        
        # XZ
        im1 = axs[1].imshow(np.mean(rho,axis=1).transpose(), cmap=colormap, interpolation='quadric', 
                origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                extent=( xe[0], xe[1], ze[0], ze[1]) )
        plt.colorbar(im1, ax=axs[1], label='Density')
        axs[1].set(xlabel=r'$X (\mu m)$', ylabel=r'$Z (\mu m)$')
        axs[1].set_xticks(xe)
        axs[1].set_yticks(ze)

        # YZ
        im2 = axs[2].imshow(np.mean(rho,axis=0).transpose(), cmap=colormap, interpolation='quadric',
                origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                extent=( ye[0], ye[1], ze[0], ze[1]) )
        plt.colorbar(im2, ax=axs[2], label='Density')
        axs[2].set(xlabel=r'$Y (\mu m)$', ylabel=r'$Z (\mu m)$')
        axs[2].set_xticks(ye)
        axs[2].set_yticks(ze)

        return fig,axs
    # }}}
    
    # PlotPlanar {{{
    def PlotPlanar2D(self, rho, bin_centers, bin_edges, **kwargs):

        # edges
        xe = [bin_edges[0][0], bin_edges[0][-1]]
        ye = [bin_edges[1][0], bin_edges[1][-1]]
        ze = [bin_edges[2][0], bin_edges[2][-1]]

        # 2D histograms (XY,XZ,YZ)
        fig,axs = plt.subplots(1,3, figsize=(12,3))

        # XY
        im0 = axs[0].imshow(np.mean(rho,axis=2).transpose(), cmap='RdYlBu_r', interpolation='quadric', 
                origin='lower', aspect='auto',
                extent=( xe[0], xe[1], ye[0], ye[1]) )
        plt.colorbar(im0, ax=axs[0])
        axs[0].set(xlabel=r'$X (\mu m)$', ylabel=r'$Y (\mu m)$')
        axs[0].set_xticks(xe)
        axs[0].set_yticks(ye)
        
        # XZ
        im1 = axs[1].imshow(np.mean(rho,axis=1).transpose(), cmap='RdYlBu_r', interpolation='quadric', 
                origin='lower', aspect='auto',
                extent=( xe[0], xe[1], ze[0], ze[1]) )
        plt.colorbar(im1, ax=axs[1])
        axs[1].set(xlabel=r'$X (\mu m)$', ylabel=r'$Z (\mu m)$')
        axs[1].set_xticks(xe)
        axs[1].set_yticks(ze)

        # YZ
        im2 = axs[2].imshow(np.mean(rho,axis=0).transpose(), cmap='RdYlBu_r', interpolation='quadric',
                origin='lower', aspect='auto',
                extent=( ye[0], ye[1], ze[0], ze[1]) )
        plt.colorbar(im2, ax=axs[2])
        axs[2].set(xlabel=r'$Y (\mu m)$', ylabel=r'$Z (\mu m)$')
        axs[2].set_xticks(ye)
        axs[2].set_yticks(ze)

        return fig,axs
    # }}}

    # PlotCylindrical {{{
    def PlotCylindrical2D(self, rho, bin_centers, bin_edges, **kwargs):

        # edges
        re = [bin_edges[0][0], bin_edges[0][-1]]
        ze = [bin_edges[1][0], bin_edges[1][-1]]

        # 2D histograms (R,Z)
        fig,ax = plt.subplots(figsize=(4,3))

        # RZ
        im = ax.imshow(rho, cmap='RdYlBu_r', interpolation='quadric', 
                origin='lower', aspect='auto',
                extent=( ze[0], ze[1], re[0], re[1]) )
        plt.colorbar(im, ax=ax)
        ax.set(xlabel='Z', ylabel='R')
        ax.set_xticks(ze)
        ax.set_yticks(re)
        
        return fig,ax
    # }}}

    # PlotSpherical {{{
    def PlotSpherical2D(self, rho, bin_centers, bin_edges, **kwargs):

        # edges
        re = [bin_edges[0][0], bin_edges[0][-1]]

        # Line plot 
        fig,ax = plt.subplots(figsize=(4,3))

        # R
        ax.plot(rho, 'bx', ms=5)
        ax.set(xlabel='R', ylabel='Local density / Global Density')
        ax.set_xticks(re)
        
        return fig,ax
    # }}}

    def Plot3D(self, rho):
        bin_edges, bin_centers = self.GetBins()
        if self.confinement_ == 'cylindrical' or self.confinement_ == 'spherical':
            print('3D plot not possible')
            return 0,0

        # edges
        xe = [bin_edges[0][0], bin_edges[0][-1]]
        ye = [bin_edges[1][0], bin_edges[1][-1]]
        ze = [bin_edges[2][0], bin_edges[2][-1]]

        # 2D histograms (XY,XZ,YZ) as contour plots
        #Setup a 3D figure and plot points as well as a series of slices
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')

        #Use one less than bin edges to give rough bin location
        X, Y, Z = np.meshgrid(bin_centers[0],bin_centers[1], bin_centers[2], indexing='ij')
        
        cpXY1 = ax1.contourf(np.mean(rho, axis=2), 
                          zdir='z', 
                          offset=ze[0], 
                          level=100, cmap=plt.cm.RdYlBu_r, alpha=1.0,
                          origin='lower', extent=( xe[0], xe[1], ye[0], ye[1])
                          )
        cpXY2 = ax1.contourf(np.mean(rho, axis=2), 
                          zdir='z', 
                          offset=ze[1], 
                          level=100, cmap=plt.cm.RdYlBu_r, alpha=1.0,
                          origin='lower', extent=( xe[0], xe[1], ye[0], ye[1])
                          )
        cpXZ1 = ax1.contourf(np.mean(rho, axis=1), 
                          zdir='y', 
                          offset=ye[0], 
                          level=100, cmap=plt.cm.RdYlBu_r, alpha=1.0,
                          origin='lower', extent=( xe[0], xe[1], ze[0], ze[1])
                          )
        cpXZ2 = ax1.contourf(X[:,0,:], np.mean(rho, axis=1), Z[:,0,:],
                          zdir='y', 
                          offset=ye[1], 
                          level=100, cmap=plt.cm.RdYlBu_r, alpha=1.0,
                          origin='lower', extent=( xe[0], xe[1], ze[0], ze[1]) )

        cpYZ1 = ax1.contourf(np.mean(rho, axis=0),
                          zdir='x', 
                          offset=xe[0], 
                          level=100, cmap=plt.cm.RdYlBu_r, alpha=1.0,
                          origin='lower', extent=( ye[0], ye[1], ze[0], ze[1]) )
        cpYZ2 = ax1.contourf(np.mean(rho, axis=0),Y[0,:,:], Z[0,:,:],
                          zdir='x', 
                          offset=xe[1], 
                          level=100, cmap=plt.cm.RdYlBu_r, alpha=1.0,
                          origin='lower', extent=( ye[0], ye[1], ze[0], ze[1]) )


        # cpXY1 = ax1.contourf(X[:,:,0],Y[:,:,0], np.mean(rho, axis=2), 
                          # zdir='z', 
                          # offset=ze[0], 
                          # level=100, 
                          # cmap=plt.cm.RdYlBu_r, 
                          # alpha=0.5)
        # cpXY2 = ax1.contourf(X[:,:,0],Y[:,:,0], np.mean(rho, axis=2), 
                          # zdir='z', 
                          # offset=ze[1], 
                          # level=100, 
                          # cmap=plt.cm.RdYlBu_r, 
                          # alpha=1.0)

        # cpXZ1 = ax1.contourf(X[:,0,:], np.mean(rho, axis=1), Z[:,0,:],
                          # zdir='y', 
                          # offset=ye[0], 
                          # level=100, 
                          # cmap=plt.cm.RdYlBu_r, 
                          # alpha=0.5)
        # cpXZ2 = ax1.contourf(X[:,0,:], np.mean(rho, axis=1), Z[:,0,:],
                          # zdir='y', 
                          # offset=ye[1], 
                          # level=100, 
                          # cmap=plt.cm.RdYlBu_r, 
                          # alpha=0.5)

        # cpYZ1 = ax1.contourf(np.mean(rho, axis=0),Y[0,:,:], Z[0,:,:],
                          # zdir='x', 
                          # offset=xe[0], 
                          # level=100, 
                          # cmap=plt.cm.RdYlBu_r, 
                          # alpha=0.5)
        # cpYZ2 = ax1.contourf(np.mean(rho, axis=0),Y[0,:,:], Z[0,:,:],
                          # zdir='x', 
                          # offset=xe[1], 
                          # level=100, 
                          # cmap=plt.cm.RdYlBu_r, 
                          # alpha=0.5)

        ax1.set_xlim(xe[0], xe[1])
        ax1.set_ylim(ye[0], ye[1])
        ax1.set_zlim(ze[0], ze[1])
        plt.colorbar(cpYZ1)
        ax1.set(xlabel=r'$X (\mu m)$', ylabel=r'$Y (\mu m)$', zlabel=r'$Z (\mu m)$')
        plt.show()
        plt.close()

        return fig,ax1
    
    # }}}
