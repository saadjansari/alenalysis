import numpy as np
import random, pdb
from matplotlib import pyplot as plt
import scipy.spatial.ckdtree
from scipy.interpolate import RectBivariateSpline
import time
from src.decorators import timer

# Plotting
def PlotRDF(FData, params, frame_avg_window=100,**kwargs):
    """ Plot the RDF for the last N frames"""
    
    frames = np.arange(FData.nframe_-frame_avg_window,FData.nframe_)
    # ends = ['minus','plus','com']
    ends = ['minus','plus']
    linestyle=['dotted','solid']

    # RDF
    print('RDF...') 
    rdf = RDF(FData.config_, **kwargs)
    
    gr = np.zeros( (len(frames), len(rdf.get_radii()), len(ends)))
    for jend,which_end in enumerate(ends):
        for idx,jframe in enumerate(frames):

            # positions
            if which_end == 'minus':
                pos = FData.pos_minus_[:,:,jframe]
            elif which_end == 'plus':
                pos = FData.pos_plus_[:,:,jframe]
            # elif which_end == 'com':
                # pos = FData.get_com()[:,:,jframe]
            gr[idx,:,jend] = rdf.Compute_v2(pos)


    fig,ax = plt.subplots()
    for jend,which_end in enumerate(ends):
        gr_end = gr[:,:,jend]
        ax.plot(rdf.get_radii(), np.mean(gr_end,axis=0), linestyle=linestyle[jend], label=which_end, color='k')
        # ax.fill_between(rdf.get_radii(),
                # np.mean(gr_end, axis=0) - np.std(gr_end, axis=0),
                # np.mean(gr_end, axis=0) + np.std(gr_end, axis=0),
                # color=cols[jend], alpha=0.2)
    # ax.set_xlim(left=-0.01)
    ax.set(xlabel=r'r / $\mu m$')
    ax.set(ylabel=r'$g(r)$', yscale='log')
    ax.set(title='RDF: Frame Window = {0}'.format(frame_avg_window))
    ax.legend()

    plt.tight_layout()
    plt.savefig(params['plot_path'] / 'rdf.pdf', bbox_inches="tight")
    plt.close()
    
    # Save data to hdf
    datapath=params['data_filestream']
    if 'filament/rdf_radii' in datapath:
        del datapath['filament/rdf_radii']
    if 'filament/rdf_plus' in datapath:
        del datapath['filament/rdf_plus']
    if 'filament/rdf_minus' in datapath:
        del datapath['filament/rdf_minus']
    datapath.create_dataset('filament/rdf_radii', data=rdf.get_radii(), dtype='f')
    datapath.create_dataset('filament/rdf_plus', data=np.mean(gr[:,:,1],axis=0), dtype='f')
    datapath.create_dataset('filament/rdf_minus', data=np.mean(gr[:,:,0],axis=0), dtype='f')

# Plotting
def PlotRDF_PAP(FData, params, frame_avg_window=100,**kwargs):
    """ Plot the RDF with c.o.m for the last N frames (split parallel and antiparallel filaments"""
    
    frames = np.arange(FData.nframe_-frame_avg_window,FData.nframe_)
    which_end='minus'

    types=['Parallel', 'Antiparallel']
    linestyle=['dotted','solid']

    # RDF
    print('RDF...') 
    rdf = RDF(FData.config_, **kwargs)
    
    gr = np.zeros( (len(frames), len(rdf.get_radii()), 2))
    for idx,jframe in enumerate(frames):

        orient = FData.orientation_[:,:,jframe]
        # positions
        if which_end == 'minus':
            pos = FData.pos_minus_[:,:,jframe]
        elif which_end == 'plus':
            pos = FData.pos_plus_[:,:,jframe]
        elif which_end == 'com':
            pos = FData.get_com()[:,:,jframe]
        gr[idx,:,0],gr[idx,:,1] = rdf.Compute_v2_PAP(pos, orient)

    fig,ax = plt.subplots()
    for jidx,jdir in enumerate(types):
        gr_dir = gr[:,:,jidx]
        ax.plot(rdf.get_radii(), np.mean(gr_dir,axis=0), linestyle=linestyle[jidx], label=jdir, color='k')
        # ax.fill_between(rdf.get_radii(),
                # np.mean(gr_end, axis=0) - np.std(gr_end, axis=0),
                # np.mean(gr_end, axis=0) + np.std(gr_end, axis=0),
                # color=cols[jend], alpha=0.2)
    # ax.set_xlim(left=-0.01)
    ax.set(xlabel=r'r / $\mu m$')
    ax.set(ylabel=r'$g(r)$', yscale='log')
    ax.set(title='RDF: Frame Window = {0}'.format(frame_avg_window))
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(params['plot_path'] / 'rdf_pap.pdf', bbox_inches="tight")
    plt.close()
    
    # Save data to hdf
    datapath=params['data_filestream']
    if 'filament/rdf_radii_pap' in datapath:
        del datapath['filament/rdf_radii_pap']
    if 'filament/rdf_p' in datapath:
        del datapath['filament/rdf_p']
    if 'filament/rdf_ap' in datapath:
        del datapath['filament/rdf_ap']
    datapath.create_dataset('filament/rdf_radii_pap', data=rdf.get_radii(), dtype='f')
    datapath.create_dataset('filament/rdf_p', data=np.mean(gr[:,:,0],axis=0), dtype='f')
    datapath.create_dataset('filament/rdf_ap', data=np.mean(gr[:,:,1],axis=0), dtype='f')

class RDF():
    # Class for computing radial density functions in a variety of geometries.
    # Confined geometries (planar, spherical, and cylindrical) are also allowed
    # unconfined geometry is also allowed
    def __init__(self, config, use_LUT=True, rcutoff=None, dr=None, rmax=None):
        self.confinement_ = config['geometry']['type'] 
        self.geometry_ = config['geometry']
        self.box_size_ = config['box_size']
        self.diameter_ = config['diameter_fil']
        self.use_LUT_ = use_LUT
        self.LUT_ = None

        # optional parameters
        if rcutoff is None:
            rcutoff = 0.7 
        if rmax is None:
            rmax = 40*self.diameter_
        if dr is None:
            dr = 0.1*self.diameter_
        self.rcutoff_ = rcutoff
        self.rmax_ = rmax
        self.dr_ = dr

        # Build LUT based on geometry
        self.BuildLUT()

    # Compute RDF {{{
    def Compute( self, c):
        """
        Calculate the rdf for single time frame
        Inputs:
            c: coorindates 3 x N (where N is number of filaments )

        Pick a (R, R+ dr) value.
        Find number of points inside that spherical shell.
        Divide by the volume of the shell
        Multiply by inverse Point density (i.e System Volume / Number of Pts)
        """
        # Radii
        radii = self.get_radii()

        # Initialize gr array
        g_r = np.zeros(shape=(len(radii)))

        # Get points number density
        num_density = self.calc_number_density(c.shape[1])

        # Ensure PBC for cKDTree
        c = self.apply_PBC(c)

        # Initialize cKDtree
        kdtree = scipy.spatial.cKDTree( c.transpose(), boxsize=self.get_special_boxsize())
        
        n_min = None        # Number of pts in smaller volume of radius r
        n_max = None        # Number of pts in larger volume of radius r+dr
        for r_idx, r in enumerate(radii):
            
            dr = self.dr_-1e-9
            # Find number of points in smaller and larger spheres
            if n_min is None:       # If n_min is not defined
                n_min = len(kdtree.query_pairs(r=r))
            n_max = len(kdtree.query_pairs(r=r+dr))
            N_shell = (n_max-n_min)

            # Ensemble averaged number for any given point
            N_shell = N_shell/c.shape[1]
            # Pairs are undercounted by factor of 2 (due to set mechanism of ckdtree)
            N_shell *= 2
            
            # Vol of shell
            vol_shell = self.get_shell_volume(c,r,r+dr)
            
            # Density in shell
            shell_density = (N_shell/vol_shell)
            
            # gr
            g_r[r_idx] = shell_density / num_density
            # print('Rmin = {0:.4f}, Rmax = {1:.4f}, Nmin = {2}, Nmax = {3}, g = {4:.2f}'.format(r,r+dr,n_min,n_max, g_r[r_idx]) )
            
            # Prep for next shell
            n_min = n_max

        return g_r
    # }}}

    # Get Shell Volume {{{
    def get_shell_volume(self,pos,r0,r1):
        # Get Shell Volume based on point positions, and geometry and 
        # pos: coorindates 3 x N (where N is number of filaments )
        # existence of LUTs
        
        if self.confinement_ == 'unconfined':
            vol_shell = self.vol_shell_unconfined(r0,r1)

        elif self.confinement_ == 'cylindrical':
            cyl_center = np.array( self.geometry_['center'] )
            cyl_radius = self.geometry_['radius'] 

            # impact parameter b
            # This is minimum distance from cylinder axis to point
            b_vals = np.linalg.norm( pos[0:2,:]-cyl_center[0:2].reshape(-1,1), 
                    axis=0)

            if self.use_LUT_:
                if self.LUT_ is None:
                    self.BuildLUT()
                vol_shell = np.mean(self.LUT_.ev(r1,b_vals) - self.LUT_.ev(r0,b_vals))
            else:
                vol_shells = [ self.vol_shell_confined_cylindrical(cyl_radius,r0,r1,b) for b in b_vals]
                vol_shell = np.mean(vol_shells)

        elif self.confinement_ == 'spherical':
            # Calculate distance from Confinement sphere center for all pts
            sph_center = np.array( self.geometry_['center'] )
            sph_radius = self.geometry_['radius'] 
            d_vals = np.linalg.norm( pos-sph_center.reshape(-1,1), axis=0)
            if self.use_LUT_:
                if self.LUT_ is None:
                    self.BuildLUT()
                vol_shell = np.mean(self.LUT_.ev(r1,d_vals) - self.LUT_.ev(r0,d_vals))
            else:
                vol_shells = [ self.vol_shell_confined_spherical(sph_radius,r0,r1,d) for d in d_vals]
                vol_shell = np.mean(vol_shells)


        elif self.confinement_ == 'planar':

            zrange = np.array( self.geometry_['range'] )
            z_vals = pos[-1,:]

            if self.use_LUT_:
                if self.LUT_ is None:
                    self.BuildLUT()
                vol_shell = np.mean(self.LUT_.ev(r1,z_vals) - self.LUT_.ev(r0,z_vals))
            else:
                vol_shells = [ self.vol_shell_confined_planar(r0,r1,
                    zrange[0], zrange[1], zz) for zz in z_vals]
                vol_shell = np.mean(vol_shells)

        return vol_shell
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
    
    @staticmethod
    def vol_shell_confined_cylindrical(R,r0,r1,b):
        v1 = RDF.vol_sphere_confined_cylindrical(R,r1,b)
        v0 = RDF.vol_sphere_confined_cylindrical(R,r0,b)
        return v1-v0
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

    def vol_shell_confined_spherical(R,r0,r1,d):
        v1 = RDF.vol_sphere_confined_spherical(R,r1,d)
        v0 = RDF.vol_sphere_confined_spherical(R,r0,d)
        return v1-v0
    # }}}

    # Volume Confined Planar Sphere/Shell {{{
    @staticmethod
    def vol_sphere_confined_planar(r, z_min, z_max, z):
        # calculated by subtracting caps
        vol_sphere_r = (4/3)*np.pi*(r**3)
        h0_min = z_min - (z-r)
        if h0_min < 0:
            h0_min = 0
        # h0_min[ h0_min < 0] = 0
        h0_max = z+r-z_max
        if h0_max < 0:
            h0_max = 0
        # h0_max[ h0_max < 0] = 0
        vol = vol_sphere_r - ( (np.pi/3)*(h0_min**2)*(3*r - h0_min)) - ( (np.pi/3)*(h0_max**2)*(3*r - h0_max))
        return vol

    def vol_shell_confined_planar(r0, r1, z_min, z_max, z):
        v1 = RDF.vol_sphere_confined_planar(r1, z_min, z_max, z)
        v0 = RDF.vol_sphere_confined_planar(r0, z_min, z_max, z)
        return v1-v0
    # }}}

    # Volume Unconfined Sphere/Shell {{{
    @staticmethod
    def vol_sphere_unconfined(r):
        return (4/3)*np.pi*(r**3)

    @staticmethod
    def vol_shell_unconfined(r0, r1):
        v1 = RDF.vol_sphere_unconfined(r1)
        v0 = RDF.vol_sphere_unconfined(r0)
        return v1-v0
    # }}}
    # }}}
    
    # LUT {{{
    def BuildLUT(self):
        # Build Look up table if required
        # speeds up computation for confinement geometries
        if not self.use_LUT_:
            return
        
        radii = self.get_radii()
        radii = np.hstack( (radii, radii[-1]+(radii[-1]-radii[-2]) ) )

        if self.confinement_ == 'cylindrical':

            R = self.geometry_['radius']
            b_vals = np.linspace(0.0,1*R,50)
            self.LUT_ = self.LUT_cylindrical(R,radii,b_vals)

        elif self.confinement_ == 'spherical':

            R = self.geometry_['radius']
            d_vals = np.linspace(0.0,1*R,50)
            self.LUT_ = self.LUT_spherical(R,radii,d_vals)

        elif self.confinement_ == 'planar':

            zrange = self.geometry_['range']
            z_vals = np.linspace(zrange[0],zrange[1],50)
            self.LUT_ = self.LUT_planar(zrange,radii,z_vals)

    @staticmethod
    def LUT_cylindrical(R,r_vals,b_vals):

        print('Building LUT...')
        data = np.zeros( (len(r_vals), len(b_vals)))

        count = 0
        for idx_r,rr in enumerate(r_vals):
            for idx_b,bb in enumerate(b_vals):
                data[idx_r,idx_b] = RDF.vol_sphere_confined_cylindrical(R,rr,bb)

                count+=1
                print('Progress = {0:.0f}%'.format(100*count/(len(r_vals)*len(b_vals))), end='\r',flush=True)

        print('LUT built successfully!')
        return RectBivariateSpline(r_vals,b_vals,data)

    @staticmethod
    def LUT_spherical(R,r_vals,d_vals):

        print('Building LUT...')
        data = np.zeros( (len(r_vals), len(d_vals)))

        count = 0
        for idx_r,rr in enumerate(r_vals):
            for idx_d,dd in enumerate(d_vals):
                data[idx_r,idx_d] = RDF.vol_sphere_confined_spherical(R,rr,dd)

                count+=1
                print('Progress = {0:.0f}%'.format(100*count/(len(r_vals)*len(d_vals))), end='\r',flush=True)

        print('LUT built successfully!')
        return RectBivariateSpline(r_vals,d_vals,data)

    @staticmethod
    def LUT_planar(zrange,r_vals,z_vals):

        print('Building LUT...')
        data = np.zeros( (len(r_vals), len(z_vals)))

        count = 0
        for idx_r,rr in enumerate(r_vals):
            for idx_z,zz in enumerate(z_vals):
                data[idx_r,idx_z] = RDF.vol_sphere_confined_planar(rr,
                        zrange[0],zrange[1],zz)

                count+=1
                print('Progress = {0:.0f}%'.format(100*count/(len(r_vals)*len(z_vals))), end='\r',flush=True)

        print('LUT built successfully!')
        return RectBivariateSpline(r_vals,z_vals,data)
    # }}}

    # Number Density {{{
    def calc_number_density(self, n_pts):
        # Calculate the number density of points in this geometry

        if self.confinement_ == 'unconfined':
            vol = np.prod(self.box_size_)
        elif self.confinement_ == 'cylindrical':
            vol = np.pi * self.box_size_[-1] * self.geometry_['radius']**2
        elif self.confinement_ == 'spherical':
            vol = (4/3) * np.pi * self.geometry_['radius']**3
        elif self.confinement_ == 'planar':
            vol = np.diff( self.geometry_['range']) * np.prod(self.box_size_[:2])
        return n_pts/vol 
    # Number Density }}}

    # Radii {{{
    def get_radii(self):
        # Figure out the radius vector to use for RDF computation
        # Range is dependent on the confinement geometry

        rcutoff = self.rcutoff_
        r_max = self.rmax_

        # if self.confinement_ == 'unconfined':
            # r_max = ( np.min(self.box_size_) / 2)*rcutoff
        # elif self.confinement_ == 'cylindrical':
            # r_max = (self.box_size_[-1]/2)*rcutoff
        # elif self.confinement_ == 'spherical':
            # r_max = 2*self.geometry_['radius']*rcutoff
        # elif self.confinement_ == 'planar':
            # r_max = ( np.min(self.box_size_[:2]) / 2)*rcutoff

        return np.arange(0.0, rcutoff*r_max, self.dr_)
    # Radii }}}

    # Special BoxSize {{{
    def get_special_boxsize(self):

        if self.confinement_ == 'unconfined':
            box_size_new = np.copy(self.box_size_)
        elif self.confinement_ == 'spherical':
            box_size_new = np.copy(100*self.box_size_)
        elif self.confinement_ == 'cylindrical':
            cyl_height = self.box_size_[2]
            box_size_new = np.array([10*cyl_height, 10*cyl_height, cyl_height])
        elif self.confinement_ == 'planar':
            box_size_new = np.array([
                self.box_size_[0], self.box_size_[1], 10*np.max(self.box_size_)
                ])
        return box_size_new
    # }}}

    # Apply PBC {{{
    def apply_PBC(self, pos):
        # pos: coorindates 3 x N (where N is number of pts)
        # Apply periodic BC
        # ensure points are within boxsize
        
        bs = self.get_special_boxsize()
        for jdim in range(pos.shape[0]):
            pos[jdim, pos[jdim,:] < 0] += bs[jdim]
            pos[jdim, pos[jdim,:] > bs[jdim]] -= bs[jdim]
        return pos
    # }}}
    
    # Compute RDF V2 {{{
    def Compute_v2( self, c):
        """
        Calculate the rdf for single time frame using a distance matrix
        Inputs:
            c: coorindates 3 x N (where N is number of filaments )

        Pick a (R, R+ dr) value.
        Find number of points inside that spherical shell.
        Divide by the volume of the shell
        Multiply by inverse Point density (i.e System Volume / Number of Pts)
        """
        # Radii
        radii = self.get_radii()

        # Initialize gr array
        g_r = np.zeros(shape=(len(radii)))

        # Get points number density
        num_density = self.calc_number_density(c.shape[1])

        # Ensure PBC for cKDTree
        c = self.apply_PBC(c)
        # Initialize cKDtree 1
        # t0 = time.time()
        kdtree1 = scipy.spatial.cKDTree( c.transpose(), boxsize=self.get_special_boxsize())
        # Initialize cKDtree 2
        kdtree2 = scipy.spatial.cKDTree( c.transpose(), boxsize=self.get_special_boxsize())

        # Bins! Add aditional entry to radii for this calculation
        bins = np.zeros(len(radii)+1)
        bins[:-1] = radii
        bins[-1] = radii[-1]+(radii[1]-radii[0])

        # Get bin counts for radii in bins using distance matrix of ckdtree
        dmat = kdtree1.sparse_distance_matrix(kdtree2, 1.1*bins[-1], output_type='coo_matrix').toarray()
        vals = dmat[np.triu_indices_from(dmat, k = 1)]
        vals[vals==0] = 100*bins[-1]
        N_shell = np.histogram(vals, bins=bins)[0]
        # ensemble averaged number for any given point
        N_shell = N_shell/c.shape[1]
        # Pairs are undercounted by factor of 2 (due to set mechanism of ckdtree)
        N_shell *= 2

        # Extract mean shell volume for each bin
        vol_shell = self.get_shell_volume_v2(c,bins)

        # Density in shell
        shell_density = (N_shell/vol_shell)
        
        # gr
        g_r = shell_density / num_density
        # print('   Time Elapsed = {0:.4f}'.format(time.time() - t0) )
        return g_r
    # }}}

    # Compute V2 PAP {{{
    def Compute_v2_PAP( self, c, orient):
        """
        Calculate the rdf for single time frame using a distance matrix
        Inputs:
            c: coorindates 3 x N (where N is number of filaments )

        Pick a (R, R+ dr) value.
        Find number of points inside that spherical shell.
        Divide by the volume of the shell
        Multiply by inverse Point density (i.e System Volume / Number of Pts)
        """
        # Radii
        radii = self.get_radii()

        # Initialize gr array
        g_r_p = np.zeros(shape=(len(radii)))
        g_r_ap = np.zeros(shape=(len(radii)))

        # Get points number density
        num_density = self.calc_number_density(c.shape[1])

        # Ensure PBC for cKDTree
        c = self.apply_PBC(c)
        # Initialize cKDtree 1
        # t0 = time.time()
        kdtree1 = scipy.spatial.cKDTree( c.transpose(), boxsize=self.get_special_boxsize())
        # Initialize cKDtree 2
        kdtree2 = scipy.spatial.cKDTree( c.transpose(), boxsize=self.get_special_boxsize())

        # Bins! Add aditional entry to radii for this calculation
        bins = np.zeros(len(radii)+1)
        bins[:-1] = radii
        bins[-1] = radii[-1]+(radii[1]-radii[0])

        # Get bin counts for radii in bins using distance matrix of ckdtree
        dmat = kdtree1.sparse_distance_matrix(kdtree2, 1.1*bins[-1], output_type='coo_matrix').toarray()
        vals = dmat[np.triu_indices_from(dmat, k = 1)]
        vals[vals==0] = 100*bins[-1]

        # Find orientaion dot products
        cosTheta = np.tensordot(orient,orient,axes=((0),(0)))
        orients_dp = cosTheta[np.triu_indices_from(cosTheta, k = 1)]

        # Split data between cosTheta > 0, cosTheta<0
        # ensemble averaged number for any given point
        N_shell_P = np.histogram(vals[orients_dp>=0], bins=bins)[0]/c.shape[1]
        N_shell_AP = np.histogram(vals[orients_dp<0], bins=bins)[0]/c.shape[1]

        # Pairs are undercounted by factor of 2 (due to set mechanism of ckdtree)
        N_shell_P *= 2
        N_shell_AP *= 2

        # Extract mean shell volume for each bin
        vol_shell = self.get_shell_volume_v2(c,bins)

        # Density in shell
        shell_density_P = (N_shell_P/vol_shell)
        shell_density_AP = (N_shell_AP/vol_shell)
        
        # gr
        g_r_p = shell_density_P / num_density
        g_r_ap = shell_density_AP / num_density
        # print('   Time Elapsed = {0:.4f}'.format(time.time() - t0) )
        return g_r_p, g_r_ap
    # }}}

    # Get Shell Volume V2 {{{
    def get_shell_volume_v2(self,pos,bins):
        # Get Shell Volume based on point positions, and geometry and 
        # pos: coorindates 3 x N (where N is number of filaments )
        
        vol_shell = np.zeros( len(bins)-1)
        if self.confinement_ == 'unconfined':
            for jr in range(len(vol_shell)):
                r0 = bins[jr]
                r1 = bins[jr+1]
                vol_shell[jr] = self.vol_shell_unconfined(bins[jr],bins[jr+1])

        elif self.confinement_ == 'cylindrical':
            cyl_center = np.array( self.geometry_['center'] )
            cyl_radius = self.geometry_['radius'] 

            # impact parameter b
            # This is minimum distance from cylinder axis to point
            b_vals = np.linalg.norm( pos[0:2,:]-cyl_center[0:2].reshape(-1,1), 
                    axis=0)

            for jr in range(len(vol_shell)):
                r0 = bins[jr]
                r1 = bins[jr+1]
                if self.use_LUT_:
                    if self.LUT_ is None:
                        self.BuildLUT()
                    vol_shell[jr] = np.mean(self.LUT_.ev(r1,b_vals) - self.LUT_.ev(r0,b_vals))
                else:
                    vol_shells = [ self.vol_shell_confined_cylindrical(cyl_radius,r0,r1,b) for b in b_vals]
                    vol_shell[jr] = np.mean(vol_shells)

        elif self.confinement_ == 'spherical':
            # Calculate distance from Confinement sphere center for all pts
            sph_center = np.array( self.geometry_['center'] )
            sph_radius = self.geometry_['radius'] 
            d_vals = np.linalg.norm( pos-sph_center.reshape(-1,1), axis=0)

            for jr in range(len(vol_shell)):
                r0 = bins[jr]
                r1 = bins[jr+1]
                if self.use_LUT_:
                    if self.LUT_ is None:
                        self.BuildLUT()
                    vol_shell[jr] = np.mean(self.LUT_.ev(r1,d_vals) - self.LUT_.ev(r0,d_vals))
                else:
                    vol_shells = [ self.vol_shell_confined_spherical(sph_radius,r0,r1,d) for d in d_vals]
                    vol_shell[jr] = np.mean(vol_shells)


        elif self.confinement_ == 'planar':

            zrange = np.array( self.geometry_['range'] )
            z_vals = pos[-1,:]

            for jr in range(len(vol_shell)):
                r0 = bins[jr]
                r1 = bins[jr+1]
                if self.use_LUT_:
                    if self.LUT_ is None:
                        self.BuildLUT()
                    vol_shell[jr] = np.mean(self.LUT_.ev(r1,z_vals) - self.LUT_.ev(r0,z_vals))
                else:
                    vol_shells = [ self.vol_shell_confined_planar(r0,r1,
                    zrange[0], zrange[1], zz) for zz in z_vals]
                    vol_shell[jr] = np.mean(vol_shells)

        return vol_shell
    # }}}

